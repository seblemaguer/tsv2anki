import pathlib
import requests
import concurrent.futures
import functools

# Image search
from PIL import Image
from ddgs import DDGS
from fastcore.all import *

# Define the timeout decorator
def timeout(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timeout = 3

        # Use ThreadPoolExecutor to run the function with a timeout
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Function '{func.__name__}' timed out after {timeout} seconds")

    return wrapper


@timeout
def search_images(term, max_images=1):
    with DDGS() as ddgs:
        search_results = ddgs.images(query=term, region="fi-fi")
        image_data = list(search_results)
        image_urls = [item.get("image") for item in image_data[:max_images]]
        return L(image_urls)


def download_image(keyword: str, cat: str, output_dir: pathlib.Path) -> pathlib.Path:
    url = str(search_images(f"+{keyword} {cat}", 1)[0])

    image_filename = output_dir / f"{keyword.replace('/', ',')}.jpg"
    if image_filename.exists():
        return image_filename

    try:
        img_data = requests.get(url).content
        img = Image.open(io.BytesIO(img_data))
        img.verify()
        with open(image_filename, "wb") as f:
            f.write(img_data)
    except Exception as e:
        raise Exception(f"Error downloading {image_filename}: {e}")

    return image_filename
