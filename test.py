import requests

text1 = {"text": "You're funny. Ugly? We're dudes on computers, moron. You're quite astonishingly stupid."}

text2 = {"text": "I've deleted the page , as we have no evidence that you are the person named on that page, and its content goes against Wikipedia's policies for the use of user pages."}

text3 = {"text": "DJ Robinson is gay as hell! he sucks his dick so much!!!!!"}

url = 'http://localhost:8080/predict'

probabilities = requests.post(url, json=text1)

print(probabilities.json())