import requests
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import json
import os
import ssl
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and model endpoint from environment variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_MODEL_ENDPOINT = os.getenv("AZURE_MODEL_ENDPOINT")

def allowSelfSignedHttps(allowed):
    # Bypass the server certificate verification on client side if needed
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


# Allow self-signed HTTPS certificates (if using a self-signed cert in your scoring service)
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
    """
    A simple token counter using whitespace split.
    In a real scenario, you may want to use the tokenizer used by your model.
    """
    return len(text.split())


def batch_html_by_tokens(html, token_limit=1000):
    """
    Splits the HTML into logical batches without cutting HTML tags.
    It parses the HTML, removes <script> tags, and groups children of the <body>
    until the token limit is reached. Each batch is then wrapped with minimal HTML.
    """
    soup = BeautifulSoup(html, 'html.parser')
    soup = remove_script_tags(soup)  # Remove <script> tags

    body = soup.body if soup.body else soup
    batches = []
    current_batch = ""

    # Iterate over the direct children of the body
    for element in body.children:
        elem_str = str(element).strip()
        if not elem_str:
            continue  # Skip empty strings or whitespace
        tokens_in_elem = count_tokens(elem_str)

        # If the element itself is larger than the token limit,
        # add it as a separate batch even if it exceeds the limit.
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

    # Wrap each batch in minimal HTML structure to ensure it's runnable
    wrapped_batches = []
    for batch in batches:
        wrapped = f"<html><head></head><body>{batch}</body></html>"
        wrapped_batches.append(wrapped)

    return wrapped_batches


def send_to_azure_model(batch, model_endpoint, api_key):
    """
    Sends a batch of HTML to the Azure AI Foundry prompt flow model using urllib.request.
    The payload includes the required fields 'html_string' and 'chat_history'.
    Expects the response JSON to contain an 'answer' field with markdown‑formatted Ruby code.
    """
    payload = json.dumps({
        "html_string": batch,
        "chat_history": [{}]
    }).encode("utf-8")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + api_key
    }
    req = urllib.request.Request(model_endpoint, data=payload, headers=headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        result_json = json.loads(result)
        answer = result_json.get("answer", "")
        # Remove markdown Ruby fences if present.
        if answer.startswith("ruby") and answer.rstrip().endswith(""):
            lines = answer.splitlines()
            answer = "\n".join(lines[1:-1])
        return answer
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
        return ""


def beautify_ruby_class(combined_code: str, site_name: str) -> str:
    """
    Merges multiple Ruby 'class MyPage ... end' snippets into one Ruby class named site_name.

    Steps:
      1. Parse out all lines inside each class block, flattening them (ignoring the actual 'class'/'end' lines).
      2. From those lines, extract:
         a) One 'include PageObject' (if any present).
         b) All accessors (e.g. text_field(...), button(...), link(...)) with no duplicates.
         c) All method definitions (def ... end), merging duplicates by method name.
            - Keep the first method's signature, append subsequent bodies to that method.
         d) Anything else is leftover lines appended at the bottom.
      3. Construct a single class definition named site_name.
    """

    # Normalize line endings
    code = combined_code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")

    # -------------------------------------------------------------------------
    # 1) Flatten all lines in class blocks, ignoring "class X" and the matching "end".
    # -------------------------------------------------------------------------
    in_class = 0
    flattened_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("class "):
            # Start of a class block
            in_class += 1
            continue  # skip writing "class ..."
        elif stripped == "end":
            if in_class > 0:
                in_class -= 1
                continue  # skip writing the 'end'
        else:
            if in_class > 0:
                flattened_lines.append(line)  # keep the line if we are inside a class

    # If no classes recognized, fallback to entire file
    if not flattened_lines:
        flattened_lines = lines[:]

    # -------------------------------------------------------------------------
    # 2) Extract a single include PageObject (if present).
    # -------------------------------------------------------------------------
    includes_found = []
    tmp_after_includes = []
    include_re = re.compile(r'^\s*include\s+PageObject\s*$')
    for line in flattened_lines:
        if include_re.match(line.strip()):
            includes_found.append(line.strip())
        else:
            tmp_after_includes.append(line)
    flattened_lines = tmp_after_includes
    found_include_pageobject = (len(includes_found) > 0)

    # -------------------------------------------------------------------------
    # 3) Extract accessors. We'll define a pattern for lines that begin with:
    #    text_field, hidden_field, button, link, element, etc.
    #    We keep them in order, but skip duplicates.
    # -------------------------------------------------------------------------
    accessor_keywords = [
        'text_field', 'hidden_field', 'button', 'link', 'select_list', 'checkbox',
        'radio_button', 'radio_button_group', 'textarea', 'div', 'span', 'table',
        'cell', 'image', 'element', 'heading', 'paragraph'
    ]
    # A line that *begins* with something like text_field( (allowing whitespace).
    accessor_pattern = re.compile(r'^\s*(?:' + "|".join(accessor_keywords) + r')\s*\(')

    accessors = []
    seen_accessors = set()
    lines_after_accessors = []
    for line in flattened_lines:
        if accessor_pattern.match(line.strip()):
            # It's an accessor line
            norm = line.strip()
            if norm not in seen_accessors:
                seen_accessors.add(norm)
                accessors.append(norm)
        else:
            lines_after_accessors.append(line)

    # -------------------------------------------------------------------------
    # 4) Extract method definitions from lines_after_accessors.
    #    We want to gather lines from "def <method_name>" up to the matching "end".
    #    We'll store them by method_name. If we see multiple definitions with the
    #    same method_name, we unify them: keep the first def ... line and last end,
    #    but merge all internal lines.
    # -------------------------------------------------------------------------
    def_pattern = re.compile(r'^\s*def\s+([a-zA-Z0-9_!?]+)')
    end_pattern = re.compile(r'^\s*end\s*$')

    # We will parse in one pass, building method blocks or leaving leftover lines alone.
    methods_extracted = {}  # {method_name -> list of [list_of_lines_for_this_definition, ...]}
    leftover = []
    in_method = False
    current_method_name = None
    current_method_lines = []

    def flush_method():
        """Store the current method block in methods_extracted, then reset."""
        nonlocal current_method_name, current_method_lines, methods_extracted
        if current_method_name is not None and current_method_lines:
            if current_method_name not in methods_extracted:
                methods_extracted[current_method_name] = []
            methods_extracted[current_method_name].append(current_method_lines)
        current_method_name = None
        current_method_lines = []

    for line in lines_after_accessors:
        # Check if this line starts a method
        start_match = def_pattern.match(line)
        if start_match and not in_method:
            # We are beginning a new method
            in_method = True
            # flush any partial
            flush_method()
            current_method_name = start_match.group(1)
            current_method_lines = [line]
        elif in_method:
            # we're inside a method, check if this line is the 'end'
            if end_pattern.match(line):
                # method is closing
                current_method_lines.append(line)
                # store it
                flush_method()
                in_method = False
            else:
                current_method_lines.append(line)
        else:
            # not in a method
            leftover.append(line)

    # If code ended while we were still in a method block, flush it:
    if in_method and current_method_lines:
        flush_method()

    # Now we unify methods that appear multiple times
    # For each method_name, we have a list of method blocks. We'll combine them:
    #   - Keep the first line from the *first* definition and the last "end" line from the first definition
    #   - Insert bodies of subsequent definitions in between (minus their 'def' and 'end').
    merged_methods = []
    for m_name, blocks in methods_extracted.items():
        if not blocks:
            continue
        # We'll start with the first block
        first_block = blocks[0]
        # We'll parse out the lines: first_block[0] is "def something",
        # last_block[-1] is "end". We'll keep them, but we'll inject in the middle any additional blocks.

        # lines for final unified method
        # keep first 'def xyz'
        final_method_lines = [first_block[0]]
        # everything except the first and last line goes in the "body"
        final_method_body = first_block[1:-1]

        # for subsequent blocks, skip the first line "def xyz" and last line "end"
        for other_block in blocks[1:]:
            # we can insert a comment to separate them if desired
            final_method_body.append("      # ----- Merged from duplicate method definition -----")
            final_method_body.extend(other_block[1:-1])

        # now close with "end" from the *first* block
        if len(first_block) > 1:
            final_method_lines.extend(final_method_body)
            final_method_lines.append(first_block[-1])
        else:
            # edge case: if the method had only "def x" but no "end"
            final_method_lines.extend(final_method_body)
            final_method_lines.append("end")

        merged_methods.append("\n".join(final_method_lines))

    # -------------------------------------------------------------------------
    # 5) Construct the final single class
    # -------------------------------------------------------------------------
    final_lines = [f"class {site_name}", ""]

    # Possibly include PageObject once
    if found_include_pageobject:
        final_lines.append("  include PageObject")
        final_lines.append("")

    # Add all unique accessors
    for ac in accessors:
        final_lines.append("  " + ac)
    final_lines.append("")

    # Add merged methods
    for method_def in merged_methods:
        # indent each line
        for line in method_def.split("\n"):
            final_lines.append("  " + line)
        final_lines.append("")

    # Finally, leftover lines at the bottom (if they are not just empty)
    leftover_stripped = [ln for ln in leftover if ln.strip()]
    if leftover_stripped:
        final_lines.append("  # leftover lines that did not belong to any recognized method or accessor")
        for ln in leftover_stripped:
            final_lines.append("  " + ln.strip())

    final_lines.append("end")

    return "\n".join(final_lines)


def beautify_ruby_file(input_file: str, output_file: str, site_name: str) -> None:
    """
    Reads a Ruby file containing multiple class definitions, beautifies it by merging
    them into one class with the given site name, and writes the result to a new file.

    Args:
      input_file (str): Path to the input Ruby file.
      output_file (str): Path where the beautified Ruby code will be saved.
      site_name (str): The new class name to use.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        combined_code = f.read()

    pretty_code = beautify_ruby_class(combined_code, site_name)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_code)

    print(f"Beautified Ruby code saved to {output_file}")


def main():
    url = input("Enter the URL: ").strip()

    try:
        html = fetch_html(url)
    except Exception as e:
        print(f"Failed to fetch HTML: {e}")
        return

    batches = batch_html_by_tokens(html, token_limit=1000)
    print(f"Total batches created: {len(batches)}")

    # Set your Azure AI Foundry prompt flow model endpoint and API key
    model_endpoint = AZURE_MODEL_ENDPOINT
    api_key = AZURE_API_KEY
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    combined_pageobject = ""
    for idx, batch in enumerate(batches, start=1):
        print(f"Processing batch {idx}/{len(batches)}...")
        pageobject = send_to_azure_model(batch, model_endpoint, api_key)
        combined_pageobject += pageobject + "\n"

    # Save the combined Ruby pageobject into a file
    combined_output = "combined_pageobject.rb"
    with open(combined_output, "w", encoding="utf-8") as f:
        f.write(combined_pageobject)
    print(f"Combined pageobject saved to {combined_output}")

    # Beautify the combined Ruby file into one class with the desired site name
    site_name = input("Enter the desired class name for the beautified file: ").strip()
    beautified_output = "beautified_pageobject.rb"
    beautify_ruby_file(combined_output, beautified_output, site_name)


if __name__ == "__main__":
    main()