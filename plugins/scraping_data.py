import requests
from bs4 import BeautifulSoup
import os
import logging
from time import sleep
from datetime import datetime

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

def fetch_data(execution_date=None,genre=None):

    url = f'https://www.goodreads.com/genres/{genre}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    result = []
    
    try:
        logger.info(f"Fetching genre page: {url}")
        response = fetch_page(url, headers)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract book URLs
        book_elements = soup.select("div.leftAlignedImage.bookBox a")
        if not book_elements:
            logger.warning("No book elements found on genre page")
            return result

        for book_element in book_elements[:35]:
            book_url = book_element.get('href', '')
            if not book_url:
                continue
                
            # Handle relative URLs
            book_url = requests.compat.urljoin('https://www.goodreads.com', book_url)
            
            logger.info(f"Processing book: {book_url}")
            try:
                book_response = fetch_page(book_url, headers)
                book_soup = BeautifulSoup(book_response.content, "html.parser")
                
                # Find review section
                review_section = book_soup.find("div", class_="ReviewsSection")
                if not review_section:
                    logger.warning(f"No reviews section found for {book_url}")
                    continue

                # Extract reviews
                review_cards = review_section.find_all("article", class_="ReviewCard")
                if not review_cards:
                    logger.warning(f"No review cards found for {book_url}")
                    continue

                for review_card in review_cards:
                    try:
                        # Extract rating
                        rating_span = review_card.find("span", class_="RatingStars RatingStars__small")
                        if not rating_span or 'aria-label' not in rating_span.attrs:
                            continue
                        
                        rating_text = rating_span['aria-label']
                        star_num = int(next(filter(str.isdigit, rating_text)))

                        # Extract comment
                        comment_span = review_card.find("span", class_="Formatted")
                        if not comment_span:
                            continue
                        
                        comment_text = comment_span.get_text(strip=True)
                        if not comment_text:
                            continue
                        comment_text = ' '.join(comment_text.split()[:51])

                        # Extract date time
                        date_span = review_card.find("span", class_="Text Text__body3")
                        
                        if not date_span:
                            continue
                        date_text = date_span.find("a").text
                        
                        date_obj = datetime.strptime(date_text, "%B %d, %Y")
                        if (execution_date != None) and (date_obj.date() != execution_date.date()):
                            continue
                        
                        # Categorize review
                        if star_num in [1, 2]:
                            result.append(("Negative", comment_text))
                        elif star_num == 3:
                            result.append(("Neutral", comment_text))
                        elif star_num in [4, 5]:
                            result.append(("Positive", comment_text))
                            
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


# def fetch_and_process_data(genres=None, **kwargs):
    
#     genres = ['art','science','history']    

#     all_data = []  
#     for genre in genres:
#         data = fetch_data(genre=genre)
#         for d in data:
#             all_data.append(d)
#     print(all_data[:10])
    



# if __name__ == "__main__":
#     try:
#         #results = fetch_data()
#         fetch_and_process_data()

#         #logger.info(f"Collected {len(results)} reviews")
#         # for i, review in enumerate(results[:5], 1):  # Show first 5 reviews
        
#         #     print(f"Review {i}: {review}")
        
#     except Exception as e:
#         print(f"Failed to complete scraping: {e}")