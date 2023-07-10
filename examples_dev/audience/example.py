#!/usr/bin/env python
"""
Suggest some relevent segments for audience targeting.
Does not consider CPM, segment size, or overlap of segments

Usage: main.py
"""

import json
import random
import time

import openai
import requests

# MediaMath audience segments base URL
AUDIENCE_SEGMENTS_URL = "https://t1.mediamath.com/api/v2.0/audience_segments"
# Your OpenAI API key
OPENAI_API_KEY = "<YOUR OPENAI KEY>"
# Your MediaMath JWT token
MEDIAMATH_API_TOKEN = "<YOUR JWT TOKEN>"
# Set the AdvertiserID
ADVERTISER_ID = 000000
# Partial GPT prompt describing what to suggest for
# example: a B2C jewelry retailer marketing products using display advertising in the United States
PROMPT = "<YOUR BRAND / CAMPAIGN DESCRIPTION>"

MEDIAMATH_API_HEADERS = {
    "Authorization": f"Bearer {MEDIAMATH_API_TOKEN}",
    "Accept": "application/vnd.mediamath.v1+json",
}

openai.api_key = OPENAI_API_KEY


def call_gpt_35(prompt, model="text-davinci-003", max_tokens=1000):
    """Takes prompt, returns response"""
    try:
        oai_response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return oai_response.choices[0].text.strip()
    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"OpenAI Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return call_gpt_35(prompt, model, max_tokens)


def reduce_segments(segment_list, result_size=10, reduce_buyable=False):
    """Takes large segment list, returns a narrowed list"""

    def json_array_to_list(string_array):
        """Takes JSON formatted string array, returns list"""
        try:
            data = json.loads(string_array)
            return data
        except ValueError:
            return []

    def score_batch(sub_list, top_size=5):
        """Takes list and runs prompt, returns narrowed list"""
        prompt = (
            f"return a JSON array of strings with the top {top_size} most "
            f"relevant segments from the following list for {PROMPT}: {sub_list}"
        )
        res = call_gpt_35(prompt)
        # print("score_batch: " + res)
        return json_array_to_list(res)

    filtered_segment_list = []
    min_depth = 3
    # take out things that we want to exclude like when the depth is not deep enough
    for t in segment_list:
        if (t[2] and reduce_buyable) or (t[3] >= min_depth and not t[2]):
            filtered_segment_list.append(t)
    # randomize list
    random.shuffle(filtered_segment_list)
    # sorted list - maybe increases breadth of final results
    # filtered_segment_list.sort(key=lambda a: a[1])
    reduced_segment_list = []
    top_size = 5
    num_after_reduce = 0
    batch_size = int(len(filtered_segment_list) / result_size * top_size)
    if batch_size > 100:
        batch_size = 100
    if batch_size > 0:
        for i in range(0, len(filtered_segment_list), batch_size):
            print(
                f">>Scoring Segments - segment_list: {len(filtered_segment_list)}, batch_size: {batch_size}"
            )
            batch = filtered_segment_list[i : i + batch_size]
            name_list = [i[1] for i in batch]
            reduced_name_list = score_batch(name_list, top_size)
            for n in reduced_name_list:
                for b in batch:
                    if b[1] == n:
                        reduced_segment_list.append(b)
        num_after_reduce = len(reduced_segment_list)
    # add back in the stuff we took out above
    for t in segment_list:
        if (not (t[2] and reduce_buyable)) and not (t[3] >= min_depth and not t[2]):
            reduced_segment_list.append(t)
    num_buyable = 0
    for t in reduced_segment_list:
        if t[2]:
            num_buyable += 1
    if num_after_reduce > 1.5 * result_size:
        return reduce_segments(reduced_segment_list, result_size, reduce_buyable)
    return reduced_segment_list, num_buyable


def get_child_segment_request_url(parent_id, advertiser_id):
    """Takes segment segment parent and advertisser, returns a URL to get children"""
    return f"{AUDIENCE_SEGMENTS_URL}?full=*&parent={parent_id}&advertiser_id={advertiser_id}"


def extract_ids(json_response):
    """Takes JSON response from API, returns a list of tuples"""
    ids_list = []
    name_list = []
    buyable_list = []
    depth_list = []
    data = json_response.get("data", [])
    for item in data:
        id_value = item.get("id")
        name_value = item.get("name")
        buyable_value = item.get("buyable")
        if id_value:
            ids_list.append(id_value)
            name_list.append(name_value)
            buyable_list.append(buyable_value)
            depth_list.append(1)
    return list(zip(ids_list, name_list, buyable_list, depth_list))


def expand_children(segment_list):
    """Takes a segment list looks up children, returns expanded list"""
    ids_list = []
    name_list = []
    buyable_list = []
    depth_list = []
    num_buyable = 0
    for t in segment_list:
        child_segment_list = []
        if not t[2]:
            request_url = get_child_segment_request_url(t[0], ADVERTISER_ID)
            child_response = requests.get(
                request_url, headers=MEDIAMATH_API_HEADERS, timeout=(3, 60)
            )
            child_segment_list = extract_ids(child_response.json())
        if len(child_segment_list) == 0:
            ids_list.append(t[0])
            name_list.append(t[1])
            buyable_list.append(t[2])
            depth_list.append(t[3])
            if t[2]:
                num_buyable += 1
        for tt in child_segment_list:
            # print (str(tt[0]) + " - " + t[1] +":" + tt[1])
            ids_list.append(tt[0])
            name_list.append(t[1] + ":" + tt[1])
            buyable_list.append(tt[2])
            depth_list.append(t[3] + tt[3])
            if tt[2]:
                num_buyable += 1
    return list(zip(ids_list, name_list, buyable_list, depth_list)), num_buyable


def main():
    """Entry point"""
    print(
        f"Using {AUDIENCE_SEGMENTS_URL} to expand segments. This will take a few minutes..."
    )
    response = requests.get(
        AUDIENCE_SEGMENTS_URL, headers=MEDIAMATH_API_HEADERS, timeout=(3, 60)
    )
    print(f"Expanding Segments - top level for advertiser {ADVERTISER_ID}")
    segment_list = extract_ids(response.json())
    last_length = -1
    num_buyable = 0
    segment_list, num_buyable = reduce_segments(segment_list, 10)
    while last_length != len(segment_list):
        last_length = len(segment_list)
        if len(segment_list) - num_buyable > 60:
            print(
                f"Narrowing Segments - num_buyable: {num_buyable}, segment_list: {len(segment_list)}"
            )
            segment_list, num_buyable = reduce_segments(segment_list, 10)
        print(
            f"Expanding Segments - num_buyable: {num_buyable}, segment_list: {len(segment_list)}"
        )
        segment_list, num_buyable = expand_children(segment_list)
    while len(segment_list) > 200:
        print(
            f"Narrowing Segments - num_buyable: {num_buyable}, segment_list: {len(segment_list)}, reduce_buyable: Yes"
        )
        segment_list, num_buyable = reduce_segments(
            segment_list, int(len(segment_list) / 6), True
        )
    print(
        f"Narrowing Segments - num_buyable: {num_buyable}, segment_list: {len(segment_list)}, reduce_buyable: Yes"
    )
    segment_list, num_buyable = reduce_segments(segment_list, 20, True)
    result = []
    for t in segment_list:
        if t[2]:
            result.append(t[1])

    out = {"suggest": PROMPT, "advertiser_id": ADVERTISER_ID, "result": result}
    json_formatted_str = json.dumps(out, indent=2)
    print(json_formatted_str)


if __name__ == "__main__":
    main()
