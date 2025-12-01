from llama_cloud_services import LlamaParse
import os
import concurrent.futures
import json

def estimate_tokens_locally(parsed_json_data):
    if isinstance(parsed_json_data, str):
        data = json.loads(parsed_json_data)
    else:
        data = parsed_json_data
    all_text = ""
    if "pages" in data:
        for page in data["pages"]:
            all_text += page.get("md", "") + "\n\n"
    else:
        all_text = str(data)
    estimated_count = len(all_text) / 3.6
    
    return int(estimated_count), all_text

def parse_single_path(path):
    api_key = os.environ.get("LLAMA_PARSE_API_KEY") 
    
    if not api_key:
        raise ValueError("LLAMA_PARSE_API_KEY environment variable not set.")
    parser = LlamaParse(
  api_key=api_key,
    language="en",
    max_pages=100,
    parse_mode="parse_page_with_agent",
    model="openai-gpt-4-1-mini",
    high_res_ocr=False,
    take_screenshot=0,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
    precise_bounding_box=True,
    )
    print(f"Submitting job for: {path}")
    result = parser.parse(file_path=path)
def remove_footer(text):
    '''
    Crawler footers contain downloaded by ...
    '''
    clean_text = re.sub(r"Downloaded by .*? on .*? CDT", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\n\s*\n", "\n\n", clean_text)
    return clean_text

def results_into_list_of_strings(results):
    list_of_strings = {}
    for key in results:
        pages = results[key][0].pages
        combined_text = ""
        number_of_pages = 1
        for page in pages:
            combined_text = "Page" + str(number_of_pages) + "\n" + combined_text + remove_footer(page.text) + "\n" 
            number_of_pages += 1
        list_of_strings[key] = [combined_text], results[key][1]
    return list_of_strings