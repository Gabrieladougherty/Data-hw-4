import requests

BASE_URL = "http://127.0.0.1:5002"

def valid_record():
    return {
        "delivery_days": 12,
        "delivery_vs_estimated": 3,
        "order_purchase_dow": 2,
        "total_price": 149.99,
        "total_freight": 25.50,
        "n_items": 1,
        "n_sellers": 1,
        "avg_price": 149.99,
        "payment_value": 175.49,
        "payment_installments": 3,
        "product_category": "electronics",
        "seller_state": "SP",
        "payment_type": "credit_card",
    }
def run_tests():
    print("Test 1: GET /health")
    r = requests.get(f"{BASE_URL}/health")
    print("Status:", r.status_code, "| Body:", r.json())

    print("\nTest 2: POST /predict (valid single)")
    r = requests.post(f"{BASE_URL}/predict", json=valid_record())
    print("Status:", r.status_code)
    print("Raw text:", r.text)

    print("\nTest 3: POST /predict/batch (5 records)")
    batch = [valid_record() for _ in range(5)]
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch)
    print("Status:", r.status_code)
    print("Raw text:", r.text)
    body = r.json()
    print("Body keys:", body.keys())
    print("Number of results:", len(body.get("results", [])))

    print("\nTest 4: POST /predict (missing required field)")
    bad = valid_record()
    del bad["total_price"]
    r = requests.post(f"{BASE_URL}/predict", json=bad)
    print("Status:", r.status_code)
    print("Raw text:", r.text)

    print("\nTest 5: POST /predict (invalid type for price)")
    bad2 = valid_record()
    bad2["total_price"] = "oops"
    r = requests.post(f"{BASE_URL}/predict", json=bad2)
    print("Status:", r.status_code)
    print("Raw text:", r.text)

if __name__ == "__main__":
    run_tests()
