import requests

print(
    requests.post(
        "http://0.0.0.0:10000",
        json={
            "message": "Would be very interesting in having a conversation. Where are you based?"
        }
    ).json()
)
