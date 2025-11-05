import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
from collections import deque
import time

BASE_URL = "https://slci.in/"
DOMAIN = urlparse(BASE_URL).netloc
OUTPUT_FILE = "data/slci_clean_data.json"

visited = set()
seen_hashes = set()
data = []

def clean_text(text):
    # Remove navigation, contact, and repeated footer lines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(Login|Register|HOME|BLOG|ABOUT|Privacy Policy|Terms & Conditions|Sitemap|WhatsApp us)+', '', text, flags=re.I)
    text = text.strip()
    return text

def extract_main_content(soup):
    """
    Extract the most relevant text block from the page.
    We’ll try several common containers used by WordPress-like sites.
    """
    selectors = [
        {"id": "main-content"},
        {"class": "content"},
        {"class": "entry-content"},
        {"class": "container"},
        {"role": "main"},
        {"id": "content"}
    ]

    for sel in selectors:
        content_div = soup.find("div", sel) or soup.find("section", sel)
        if content_div:
            text = content_div.get_text(separator=" ", strip=True)
            if len(text.split()) > 50:
                return clean_text(text)

    # fallback: get all paragraph text
    text = " ".join([p.get_text(separator=" ", strip=True) for p in soup.find_all("p")])
    return clean_text(text)

def scrape_page(url):
    try:
        print(f"🕸️ Scraping: {url}")
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        title = soup.title.string.strip() if soup.title else ""
        content = extract_main_content(soup)
        if not content or len(content) < 100:
            print(f"⚠️ Skipping {url}: No meaningful text found")
            return None

        # Avoid duplicate content
        hash_val = hash(content)
        if hash_val in seen_hashes:
            print(f"⚠️ Duplicate page: {url}")
            return None
        seen_hashes.add(hash_val)

        page_data = {"title": title, "url": url, "content": content}
        data.append(page_data)
        return soup

    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

def crawl_website(base_url, max_pages=50):
    queue = deque([base_url])

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        soup = scrape_page(url)
        if not soup:
            continue

        # find new links
        for link_tag in soup.find_all("a", href=True):
            href = link_tag["href"]
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # stay inside domain
            if parsed.netloc == DOMAIN and full_url not in visited:
                # filter out fragments, query params, and media files
                if not any(full_url.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".jpeg", ".gif"]):
                    if "#" not in full_url and "?" not in full_url:
                        queue.append(full_url)
        time.sleep(0.5)

    # save data
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Done. Scraped {len(data)} unique pages → saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    crawl_website(BASE_URL, max_pages=50)
