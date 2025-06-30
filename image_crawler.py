import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin, urlparse

"""
주어진 URL에서 이미지를 찾아 다운로드합니다.
Args:
    url (str): 이미지를 다운로드할 웹페이지의 URL.
    download_dir (str): 이미지를 저장할 디렉토리 이름.
    delay_seconds (int): 각 요청 사이에 기다릴 시간 (초).
    max_images (int): 다운로드할 최대 이미지 개수. None이면 제한 없음.
Returns:
    int: 다운로드된 이미지의 개수. 오류 발생 시 -1 반환.
"""
def download_images_from_url(url, download_dir='downloaded_images', delay_seconds=1, max_images=None, headers=None):
    downloaded_images = 0
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    print(f"[{url}]에서 이미지 수집 시작...")

    if headers is None:
        headers = {
            'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        }

    try:
        response = requests.get(url, timeout=10, headers=headers) # 10초 타임아웃 설정
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
    except requests.exceptions.RequestException as e:
        print(f"URL 접속 오류: {e}")
        return -1

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    for img_tag in img_tags:
        if max_images is not None and downloaded_images >= max_images:
            print(f"최대 이미지 수집 개수 ({max_images})에 도달하여 중단합니다.")
            break
        img_url = img_tag.get('src') # 'src' 속성에서 이미지 URL 가져오기
        if not img_url:
            continue

        # 상대 경로인 경우 절대 경로로 변환
        img_url = urljoin(url, img_url)

        # 유효한 이미지 확장자만 다운로드 (선택 사항)
        if not (img_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))):
            continue

        try:
            img_name = os.path.basename(urlparse(img_url).path) # URL에서 파일 이름 추출
            if not img_name: # 파일 이름이 비어있는 경우 스킵
                continue

            img_path = os.path.join(download_dir, img_name)

            # 이미지가 이미 존재하는 경우 건너뛰기
            if os.path.exists(img_path):
                print(f"  이미 존재하여 건너뜀: {img_name}")
                continue

            img_response = requests.get(img_url, stream=True, timeout=10)
            img_response.raise_for_status()

            with open(img_path, 'wb') as f:
                for chunk in img_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  다운로드 완료: {img_name}")
            downloaded_images += 1

        except requests.exceptions.RequestException as e:
            print(f"  이미지 다운로드 오류 ({img_url}): {e}")
        except Exception as e:
            print(f"  알 수 없는 오류 ({img_url}): {e}")

        time.sleep(delay_seconds) # 서버에 부담을 주지 않기 위해 지연 시간 추가

    print(f"[{url}] 이미지 수집 완료.")
    return downloaded_images