FROM ubuntu
RUN apt update
RUN apt install nodejs npm -y --fix-missing

WORKDIR /playwright
RUN npm install playwright @playwright/mcp@latest -y
RUN npx playwright install-deps
RUN npx playwright install firefox

EXPOSE 8931
CMD npx -y "@playwright/mcp" --isolated --port 8931 --browser firefox