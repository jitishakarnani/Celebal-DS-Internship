import os
import time
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join("docs_corpus", "official_docs")

SOURCES = {
    "pandas_10min": "https://pandas.pydata.org/docs/user_guide/10min.html",
    "pandas_basics": "https://pandas.pydata.org/docs/user_guide/basics.html",
    "pandas_groupby": "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "pandas_merging": "https://pandas.pydata.org/docs/user_guide/merging.html",
    "pandas_missing_data": "https://pandas.pydata.org/docs/user_guide/missing_data.html",
    "pandas_indexing": "https://pandas.pydata.org/docs/user_guide/indexing.html",
    "pandas_reshaping": "https://pandas.pydata.org/docs/user_guide/reshaping.html",
    "python_exceptions": "https://docs.python.org/3/library/exceptions.html",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ds-copilot-docs-fetch/1.0)"}


def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    main = (
        soup.find("article", attrs={"role": "main"})
        or soup.find("div", class_="bd-article")
        or soup.find("main")
        or soup.find("div", attrs={"id": "main-content"})
        or soup.body
    )
    if main is None:
        return ""
    for a in main.find_all("a", class_="headerlink"):
        a.decompose()
    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def fetch_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, url in SOURCES.items():
        print(f"Fetching {name} <- {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  FAILED: {e}")
            continue
        text = extract_main_text(resp.text)
        if not text.strip():
            print("  WARNING: no content extracted, skipping")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"SOURCE: {url}\n\n")
            f.write(text)
        print(f"  Saved -> {out_path} ({len(text)} chars)")
        time.sleep(1)


if __name__ == "__main__":
    fetch_all()
