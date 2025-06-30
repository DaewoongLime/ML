from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import time
import os
from urllib.parse import urljoin, urlparse

def download_images_with_selenium(url, download_dir='downloaded_images', delay_seconds=2, max_images=None):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Chrome 옵션 설정
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # GUI 없이 백그라운드에서 실행 (선택 사항, 개발 시에는 주석 처리하여 브라우저 동작 확인)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # 실제 브라우저 User-Agent를 설정 
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

    # WebDriver 초기화
    # chromedriver.exe 파일이 현재 스크립트와 같은 폴더에 있거나, PATH에 설정되어 있으면 Service 객체 없이 바로 가능:
    driver = webdriver.Chrome(options=chrome_options)
    # 만약 특정 경로에 있다면:
    # service = Service(executable_path='./path/to/your/chromedriver.exe')
    # driver = webdriver.Chrome(service=service, options=chrome_options)


    downloaded_count = 0
    print(f"[{url}]에서 Selenium을 사용하여 이미지 수집 시작...")

    try:
        driver.get(url) # 웹페이지 로드 (여기서 JS 실행 및 챌린지 통과 시도)
        print(f"페이지 로딩 중... {delay_seconds*2}초 대기")
        time.sleep(delay_seconds * 2) # 페이지가 완전히 로드되고 JavaScript가 실행될 충분한 시간 대기

        # JavaScript에 의해 동적으로 로드되는 콘텐츠가 있다면, 스크롤을 내리거나 특정 요소를 클릭하는 등 추가 동작이 필요할 수 있습니다.
        # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # 페이지 끝까지 스크롤 예시

        # 완전히 렌더링된 페이지 소스 가져오기
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        img_tags = soup.find_all('img')

        for img_tag in img_tags:
            if max_images is not None and downloaded_count >= max_images:
                print(f"최대 다운로드 수 ({max_images}개)에 도달하여 중지합니다.")
                break

            img_url = img_tag.get('src') or img_tag.get('data-src') # lazy loading 대비 data-src도 확인
            if not img_url:
                continue

            img_url = urljoin(url, img_url)

            if not (img_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))):
                continue

            try:
                img_name = os.path.basename(urlparse(img_url).path)
                if not img_name:
                    continue

                img_path = os.path.join(download_dir, img_name)

                if os.path.exists(img_path):
                    print(f"  이미 존재하여 건너뜀: {img_name}")
                    continue

                # Selenium을 사용했으므로 requests.get() 대신 실제 브라우저가 직접 이미지 파일을 다운로드하도록 할 수 있습니다.
                # 하지만 편의상 이미지 URL을 얻은 후 다시 requests를 사용하는 경우가 많습니다.
                # 이 때, Selenium 세션의 쿠키를 requests에 전달해주는 것이 중요할 수 있습니다.
                # (간단하게는 User-Agent만 맞춰줘도 될 때도 있습니다.)

                # Selenium 세션의 쿠키를 requests로 전달
                cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
                img_headers = {
                    'User-Agent': chrome_options.arguments[-1].split('=')[1], # Selenium에 설정한 User-Agent 사용
                    'Referer': url # 이미지의 Referer는 이미지 요청을 보낸 페이지 (현재 URL)
                }
                img_response = requests.get(img_url, stream=True, timeout=10, headers=img_headers, cookies=cookies)
                img_response.raise_for_status()

                with open(img_path, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  다운로드 완료: {img_name}")
                downloaded_count += 1

            except Exception as e:
                print(f"  이미지 처리 오류 ({img_url}): {e}")

            time.sleep(delay_seconds) # 이미지 다운로드 사이 딜레이

    except Exception as e:
        print(f"Selenium 페이지 로드 또는 파싱 오류: {e}")
    finally:
        driver.quit() # 브라우저 인스턴스 닫기 (매우 중요!)

    print(f"[{url}] 이미지 수집 완료. 총 {downloaded_count}개 다운로드.")
    return downloaded_count