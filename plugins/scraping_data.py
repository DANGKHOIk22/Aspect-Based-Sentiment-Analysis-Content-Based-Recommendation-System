import requests
from bs4 import BeautifulSoup
import os,sys
import logging
from time import sleep
from datetime import datetime
import re
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_page(url, headers, retries=3, delay=2):
    """Helper function to fetch a page with retry logic"""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                sleep(delay)
            continue
    raise requests.RequestException(f"Failed to fetch {url} after {retries} attempts")

def fetch_data(idx,genre, num_books = 15):

    url = f'https://www.goodreads.com/genres/{genre}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    result = []
    image_dir = os.path.join(PROJECT_ROOT, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    try:
        logger.info(f"Fetching genre page: {url}")
        response = fetch_page(url, headers)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract book URLs
        book_elements = soup.select("div.leftAlignedImage.bookBox a")
        if not book_elements:
            logger.warning("No book elements found on genre page")
            return result

        for book_ids,book_element in enumerate(book_elements[:num_books]):
            book_url = book_element.get('href', '')
            if not book_url:
                continue
                
            # Handle relative URLs
            book_url = requests.compat.urljoin('https://www.goodreads.com', book_url)
            
            logger.info(f"Processing book: {book_url}")
            try:
                book_response = fetch_page(book_url, headers)
                book_soup = BeautifulSoup(book_response.content, "html.parser")
                image_tag = book_soup.find("img", class_="ResponsiveImage")
                if image_tag and image_tag.get("src"):
                    image_url = image_tag["src"]
                response = requests.get(image_url)
                image_name = f'{idx*num_books + book_ids}.jpg'
                path = os.path.join(image_dir, image_name)
                if response.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(response.content)
                title_tag = book_soup.find("h1", {"data-testid": "bookTitle"})
                book_title = title_tag.get_text(strip=True) if title_tag else None

                details_section = book_soup.find("div", class_="BookDetails")
                pages = None

                if details_section:
                    pages_format_tag = details_section.find("p", {"data-testid": "pagesFormat"})
                    if pages_format_tag:
                        pages_text = pages_format_tag.get_text(strip=True)

                try:
                    if pages_text:
                        pages = int(pages_text.split()[0])
                except ValueError:
                    pages = None
                
                # Find review section
                review_section = book_soup.find("div", class_="ReviewsSection")
                if not review_section:
                    logger.warning(f"No reviews section found for {book_url}")
                    continue
                # Extract overall rating
                rating_tag = review_section.find("div", class_="RatingStatistics__rating")
                rating = float(rating_tag.get_text(strip=True)) if rating_tag else None

                # Extract ratings and reviews count
                meta_div = review_section.find("div", class_="RatingStatistics__meta")
                if meta_div:
                    ratings_span = meta_div.find("span", {"data-testid": "ratingsCount"})
                    reviews_span = meta_div.find("span", {"data-testid": "reviewsCount"})

                    if ratings_span:
                        ratings_text = ratings_span.get_text(strip=True)
                        match = re.search(r'\d+', ratings_text)
                        ratings_count = int(match.group()) if match else 0
                    else:
                        ratings_count = 0
                    if reviews_span:
                        reviews_text = reviews_span.get_text(strip=True)
                        match = re.search(r'\d+', reviews_text)
                        reviews_count = int(match.group()) if match else 0
                    else:
                        reviews_count = 0
                else:
                    ratings_count = 0
                    reviews_count = 0


                # Extract reviews
                review_cards = review_section.find_all("article", class_="ReviewCard")
                if not review_cards:
                    logger.warning(f"No review cards found for {book_url}")
                    continue

                for review_card in review_cards[:10]:
                    try:
                        # Extract comment
                        comment_span = review_card.find("span", class_="Formatted")
                        if not comment_span:
                            continue
                        
                        comment_text = comment_span.get_text(strip=True)
                        if not comment_text:
                            continue
                        comment_text = ' '.join(comment_text.split()[:51])

                        
                        result.append((idx*num_books + book_ids,book_title,genre, pages,rating, ratings_count, reviews_count, comment_text))
   
                            
                    except (ValueError, AttributeError) as e:
                        logger.error(f"Error processing review in {book_url}: {e}")
                        continue
                sleep(1)
                
            except requests.RequestException as e:
                logger.error(f"Error fetching book page {book_url}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Unexpected error during scraping: {e}")
        raise
        
    return result



