from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs, urlencode
import time

# Initialize WebDriver for Firefox
gecko_driver_path = "C:\\Users\\jasar\\Desktop\\geckodriver-docker-image-120.0.1-driver0.33.0-r0"
service = Service(gecko_driver_path)
driver = webdriver.Firefox(service=service)

driver.maximize_window()

def get_next_page_url(current_url):
    print(f"Current URL: {current_url}")  # Debugging output
    parsed_url = urlparse(current_url)
    query_params = parse_qs(parsed_url.query)
    current_pos = int(query_params.get('frm_pos', [1])[0])
    maxrec = int(query_params.get('MAXREC', [10])[0])
    next_pos = current_pos + maxrec

    print(f"Next Position: {next_pos}")  # Debugging output

    max_frm_pos = 500.  # Adjust as needed
    if next_pos > max_frm_pos:
        return None

    query_params['frm_pos'] = [str(next_pos)]
    new_query_string = urlencode(query_params, doseq=True)
    next_page_url = parsed_url._replace(query=new_query_string).geturl()
    print(f"Next Page URL: {next_page_url}")  # Debugging output
    return next_page_url

processed_urls = set()  # Initialize an empty set to keep track of processed URLs

def process_page(url):
    if url in processed_urls:
        return

    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//body")))

    processed_links = set()

    while True:
        prodaja_links = driver.find_elements(By.XPATH, "//a[contains(text(), 'sklep o prodaji')]")
        if not prodaja_links:
            break

        for index in range(len(prodaja_links)):
            prodaja_links = driver.find_elements(By.XPATH, "//a[contains(text(), 'sklep o prodaji')]")
            if index >= len(prodaja_links):
                break  # Avoid stale element reference

            link = prodaja_links[index]
            href = link.get_attribute('href')
            if href and href not in processed_links:
                processed_links.add(href)

                driver.execute_script("arguments[0].scrollIntoView();", link)
                driver.execute_script("arguments[0].click();", link)

                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Vsebina procesnega dejanja')]"))).click()

                main_window = driver.current_window_handle
                time.sleep(2)
                if len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    time.sleep(2)
                    driver.close()

                driver.switch_to.window(main_window)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//body")))

                print("Returned to main window, continuing with next link")  # Debugging output

    processed_urls.add(url)

# Rest of your script...




# Start the process

start_url = "https://www.ajpes.si/eObjave/rezultati.asp?podrobno=0&tipdolznika=-1&tippostopka=-1&id_skupinavrsta=51&id_skupinapodvrsta=-1&sodisce=-1&datumdejanja_od=18.12.2023&maxrec=10&id_skupina=51&frm_pos=1"  # "https://www.ajpes.si/eObjave/rezultati.asp?podrobno=0&id_skupina=51&TipDolznika=-1&TipPostopka=-1&id_SkupinaVrsta=-1&id_skupinaPodVrsta=-1&Dolznik=&Oblika=&MS=&DS=&StStevilka=&Sodisce=-1&DatumDejanja_od=&DatumDejanja_do=&sys_ZacetekObjave_od=&sys_ZacetekObjave_do=&MAXREC=10"

current_url = start_url
try:
    while current_url and current_url not in processed_urls:
        process_page(current_url)
        current_url = get_next_page_url(current_url)
finally:
    driver.quit()

