from fastapi import FastAPI, HTTPException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from PIL import Image
import io
import base64

def screenshot(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
       
        driver = webdriver.Chrome(options=chrome_options)

        driver.get(url)
        
        # Zoom out 200%
        driver.execute_script("document.body.style.zoom='50%'")
        
        # Optional: Wait for the zoom to take effect
        driver.implicitly_wait(2)
        
        # Set the viewport size to the full page height and width
        driver.execute_script("window.scrollTo(0, 0);")
        driver.set_window_size(1920, 1080)  # Adjust size as needed
        
        # Scroll to bottom to ensure the page is fully loaded
        body = driver.find_element(By.TAG_NAME, 'body')
        driver.execute_script("arguments[0].scrollTo(0, arguments[0].scrollHeight);", body)
        
        # Take a full-page screenshot
        screenshot = driver.get_screenshot_as_base64()
        driver.quit()

        return screenshot
    
    except Exception as e:
        driver.quit()
        raise HTTPException(status_code=500, detail=str(e))
