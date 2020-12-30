import json
import random
import os


input_file = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/data/1_multiwoz/restaurant_db.json'
output_file = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/data/1_multiwoz/restaurant_db_transformed.json'
fout = open(output_file, 'w')
rest_dict = {"data": []}

rest_domain = ['restaurant-dining-room', 'restaurant-kid-friendly', 'restaurant-riverside', 'restaurant-rooftop', 'restaurant-rooftop-bar', 'restaurant-with-balcony', 'restaurant-with-bar', 'restaurant-with-couches', 'restaurant-with-dance-floor', 'restaurant-with-garden', 'restaurant-with-private-room', 'restaurant-with-sea-view']
images_consume_history = []
images_path = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/images/restaurant/'

dict = {
            "title": "University_of_Notre_Dame",
            "paragraphs": [
                {
                    "context": "",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": "",
                                    "text": ""
                                }
                            ],
                            "question": "",
                            "id": ""
                        }
                    ]
                }
            ]
        }

with open(input_file) as fin:
    f = json.load(fin)
    for elm in f:
        address = elm['address']
        address = '_'.join(address.split(" "))
        area = elm['area']
        cuisine = elm['food']
        cuisine = "_".join(cuisine.split(" "))
        name = elm['name']
        name = '_'.join(name.split(" "))
        price_range = elm['pricerange']
        images = []

        sampled_image_types = random.sample(rest_domain, 2)
        for type in sampled_image_types:
            file_names = []
            path = images_path + type
            for parent, dirnames, filenames in os.walk(path):
                file_names = filenames
            while True:
                x = random.randint(0, len(filenames) - 1)
                file_name = file_names[x]
                image_path = path + '/' + file_name
                if image_path not in images_consume_history:
                    break
            images.append('http://10.100.231.7/' + type + '/' + file_name)
            images_consume_history.append(image_path)
        image_1 = images[0]
        image_2 = images[1]

        context = "\n" + name + " address " + address + "\n" + name + " area " + area + "\n" + name + " cuisine " + cuisine + "\n" + name + " pricerange " + price_range + "\n" + name + " image1 " + image_1 + "\n" + name + " image2 " + image_2
        dict = {
            "title": "University_of_Notre_Dame",
            "paragraphs": [
                {
                    "context": "",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": "",
                                    "text": ""
                                }
                            ],
                            "question": "",
                            "id": ""
                        }
                    ]
                }
            ]
        }
        dict["paragraphs"][0]["context"] = context
        rest_dict["data"].append(dict)

json.dump(rest_dict, fout)
