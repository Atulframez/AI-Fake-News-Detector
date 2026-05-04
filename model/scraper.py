"""
Article Scraper — Extract news text from any URL.

Uses trafilatura (best-in-class article extractor) with
BeautifulSoup as fallback for maximum coverage.
"""

import re
import requests
from urllib.parse import urlparse

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Cache-Control': 'no-cache',
}

TIMEOUT = 15  # seconds
MAX_TEXT_LENGTH = 50_000  # chars — guard against giant pages


class ArticleScraper:
    """
    Extract clean article text from any news URL.

    Priority:
        1. trafilatura (state-of-the-art boilerplate removal)
        2. BeautifulSoup paragraph extraction (fallback)
    """

    def scrape(self, url: str) -> dict:
        """
        Scrape article text from a URL.

        Returns:
            {
                'success': bool,
                'url': str,
                'domain': str,
                'text': str,
                'title': str,
                'error': str or None,
                'method': str,  # 'trafilatura' | 'beautifulsoup' | 'failed'
            }
        """
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        domain = self._extract_domain(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            html = resp.text
        except requests.exceptions.Timeout:
            return self._fail(url, domain, "Request timed out after 15 seconds.")
        except requests.exceptions.ConnectionError:
            return self._fail(url, domain, "Could not connect to the URL. Check if it's valid.")
        except requests.exceptions.HTTPError as e:
            return self._fail(url, domain, f"HTTP error: {e}")
        except Exception as e:
            return self._fail(url, domain, str(e))

        # 1. Try trafilatura
        if HAS_TRAFILATURA:
            result = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
            )
            title = self._extract_title_bs4(html) if HAS_BS4 else ''
            if result and len(result.strip()) > 100:
                return {
                    'success': True,
                    'url': url,
                    'domain': domain,
                    'text': result.strip(),
                    'title': title,
                    'error': None,
                    'method': 'trafilatura',
                    'word_count': len(result.split()),
                }

        # 2. Fallback: BeautifulSoup
        if HAS_BS4:
            text, title = self._bs4_extract(html)
            if text and len(text.strip()) > 100:
                return {
                    'success': True,
                    'url': url,
                    'domain': domain,
                    'text': text.strip(),
                    'title': title,
                    'error': None,
                    'method': 'beautifulsoup',
                    'word_count': len(text.split()),
                }

        return self._fail(url, domain, "Could not extract article text from this page.")

    def _bs4_extract(self, html: str) -> tuple[str, str]:
        """Extract text from paragraphs using BeautifulSoup.

        Removes noisy tags (script, nav, footer, etc.) before collecting
        paragraph text, then collapses whitespace.
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                         'aside', 'form', 'noscript', 'iframe']):
            tag.decompose()

        title = self._extract_title_bs4(html)

        # Try article tag first, then body
        article = soup.find('article')
        container = article if article else soup.find('body')
        if not container:
            return '', title

        paragraphs = container.find_all('p')
        text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
        text = re.sub(r'\s+', ' ', text).strip()
        return text, title

    def _extract_title_bs4(self, html: str) -> str:
        """Extract page title."""
        if not HAS_BS4:
            return ''
        try:
            soup = BeautifulSoup(html, 'html.parser')
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title['content'].strip()
            if soup.title:
                return soup.title.get_text(strip=True)
        except Exception:
            pass
        return ''

    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '').lower()
            return domain or url
        except Exception:
            return url

    def is_supported_scheme(self, url: str) -> bool:
        """Return True only for http/https URLs."""
        return url.lower().startswith(('http://', 'https://'))

    def _fail(self, url: str, domain: str, error: str) -> dict:
        return {
            'success': False,
            'url': url,
            'domain': domain,
            'text': '',
            'title': '',
            'error': error,
            'method': 'failed',
            'word_count': 0,
        }
