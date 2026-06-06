# -*- coding: utf-8 -*-
"""
Automated Image Scraper Pipeline (Bing Images)
"""

import os
import time
import requests
import yaml
import hashlib
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

class ImageScraper:
    def __init__(self, config_path="config/gan_config.yaml"):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            self.config = full_config.get('scraper', full_config)
            
        self.download_path = Path(self.config.get('raw_image_root', 'data/raw'))
        os.makedirs(self.download_path, exist_ok=True)
        self.setup_browser()
        
    def setup_browser(self):
        print("Initializing headless browser...")
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
    def scrape_images(self, custom_search_terms=None):
        search_terms = custom_search_terms or self.config.get('search_terms', ['faces', 'landscapes'])
        max_total = self.config.get('max_total_images', 100)
        per_term = self.config.get('num_images_per_term', 50)
        
        downloaded_images = []
        
        for term in search_terms:
            if len(downloaded_images) >= max_total:
                break
            term_paths = self._download_images_for_term(term, per_term, max_total - len(downloaded_images))
            downloaded_images.extend(term_paths)
            
        return downloaded_images

    def _download_images_for_term(self, term, num_images, capacity):
        downloaded_paths = []
        target_count = min(num_images, capacity)
        
        if target_count <= 0: 
            return downloaded_paths
        
        try:
            print(f"Searching Bing Images for: '{term}'...")
            self.driver.get(f"https://www.bing.com/images/search?q={term}")
            time.sleep(2)
            
            # Scroll to load dynamic images
            for _ in range(3):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
                
            # Bing uses the 'mimg' class for its main grid images
            elements = self.driver.find_elements(By.CSS_SELECTOR, "img.mimg")
            print(f"Found {len(elements)} image nodes. Beginning extraction...")
            
            for thumb in elements:
                if len(downloaded_paths) >= target_count: 
                    break
                try:
                    # Snag the source URL (Bing sometimes uses data-src for lazy loading)
                    src = thumb.get_attribute('src') or thumb.get_attribute('data-src')
                    if src and src.startswith('http'):
                        path = self._download_image(src, term)
                        if path: 
                            downloaded_paths.append(path)
                except Exception as e: 
                    continue
                    
        except Exception as e:
            print(f"Scraping failed for '{term}': {e}")
            
        return downloaded_paths
        
    def _download_image(self, url, term):
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            filename = f"{term.replace(' ', '_')}_{url_hash[:10]}.jpg"
            filepath = self.download_path / filename
            
            if filepath.exists(): 
                return str(filepath)
                
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            res = requests.get(url, timeout=5, headers=headers)
            res.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(res.content)
                
            return str(filepath) if filepath.stat().st_size > 0 else None
        except:
            return None
            
    def close(self):
        if hasattr(self, 'driver'): 
            self.driver.quit()
        
    def __del__(self):
        self.close()

if __name__ == "__main__":
    scraper = ImageScraper()
    results = scraper.scrape_images()
    print(f"✓ Operation Complete. Downloaded {len(results)} scraped images total.")