import requests
from bs4 import BeautifulSoup

url = "https://www.espn.com/espn/sitemap/_/sport/mlb"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

links = []

for a in soup.find_all("a"):
    href = a.get("href")
    if href and "/mlb/story/" in href:
        links.append(href)

links = list(set(links))

print("Articles found:", len(links))

for link in links[:10]:
    print(link)