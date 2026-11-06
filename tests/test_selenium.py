import urllib.request

import pytest


def api_disponible():
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=3)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not api_disponible(), reason="API non demarree sur localhost:8000"
)


def test_application():
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("http://localhost:8000/docs")
        assert "Churn Prediction API" in driver.title
    finally:
        driver.quit()
