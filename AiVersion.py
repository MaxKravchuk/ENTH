import requests
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import json
import os
import ssl
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and model endpoint from environment variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_MODEL_ENDPOINT = os.getenv("AZURE_MODEL_ENDPOINT")

if not AZURE_API_KEY or not OPENAI_API_KEY or not AZURE_MODEL_ENDPOINT:
    raise Exception("Missing required environment variables. Check your .env file.")


def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(True)


def fetch_html(url):
    """Fetch full HTML from the provided URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def remove_script_tags(soup):
    """Remove all <script> tags from the BeautifulSoup object."""
    for script in soup.find_all('script'):
        script.decompose()
    return soup


def count_tokens(text):
    return len(text.split())


def batch_html_by_tokens(html, token_limit=1000):
    """Splits HTML into logical parts ensuring each batch fits within the token limit."""
    soup = BeautifulSoup(html, 'html.parser')
    soup = remove_script_tags(soup)

    body = soup.body if soup.body else soup
    batches = []
    current_batch = ""

    for element in body.children:
        elem_str = str(element).strip()
        if not elem_str:
            continue
        tokens_in_elem = count_tokens(elem_str)

        if tokens_in_elem > token_limit:
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            batches.append(elem_str)
        else:
            current_tokens = count_tokens(current_batch)
            if current_tokens + tokens_in_elem <= token_limit:
                current_batch += "\n" + elem_str
            else:
                batches.append(current_batch)
                current_batch = elem_str

    if current_batch:
        batches.append(current_batch)

    wrapped_batches = [f"<html><head></head><body>{batch}</body></html>" for batch in batches]
    return wrapped_batches


def send_to_azure_model(batch):
    """Sends HTML batch to Azure AI Foundry model."""
    payload = json.dumps({
        "html_string": batch,
        "chat_history": [{}]
    }).encode("utf-8")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {AZURE_API_KEY}'
    }

    req = urllib.request.Request(AZURE_MODEL_ENDPOINT, data=payload, headers=headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        result_json = json.loads(result)
        return result_json.get("answer", "")
    except urllib.error.HTTPError as error:
        print("Request failed with status code:", error.code)
        return ""


def beautify_ruby_with_openai(ruby_code):
    """Sends Ruby code to OpenAI API to be properly formatted and beautified."""
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are an expert Ruby developer. Format and improve the readability of the following Ruby "
                        "code:"},
            {"role": "user", "content": ruby_code}
        ]
    )

    return completion.choices[0].message.content


def main():
    url = input("Enter the URL: ").strip()

    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"Failed to fetch HTML: {e}")
        return

    batches = batch_html_by_tokens(html, token_limit=1000)
    print(f"Total batches created: {len(batches)}")

    combined_pageobject = ""
    for idx, batch in enumerate(batches, start=1):
        print(f"Processing batch {idx}/{len(batches)}...")
        pageobject = send_to_azure_model(batch)
        combined_pageobject += pageobject + "\n"

    print("Sending combined Ruby code to OpenAI for beautification...")
    beautified_ruby = beautify_ruby_with_openai(combined_pageobject)

    beautified_output = "beautified_pageobject.rb"
    with open(beautified_output, "w", encoding="utf-8") as f:
        f.write(beautified_ruby)

    print(f"Beautified Ruby code saved to {beautified_output}")


if __name__ == "__main__":
    main()