from faker import Faker

Faker.seed(23)


def generate_random_text(num_sentences):
    fake = Faker()
    text = ""
    for _ in range(num_sentences):
        text += fake.sentence() + " "
    return text
