# Note: Since output width & height need to be divisible by 8, the w & h -values do
#       not exactly match the stated aspect ratios... but they are "close enough":)

aspect_ratio_list = [
    {
        "name": "1024_1024",
        "w": 1024,
        "h": 1024,
    },
    {
        "name": "1024_680",
        "w": 1024,
        "h": 680,
    },
    {
        "name": "680_1024",
        "w": 680,
        "h": 1024,
    },
    {
        "name": "1024_768",
        "w": 1024,
        "h": 768,
    },
    {
        "name": "768_1024",
        "w": 768,
        "h": 1024,
    },
    {
        "name": "1024_576",
        "w": 1024,
        "h": 576,
    },
    {
        "name": "576_1024",
        "w": 576,
        "h": 1024,
    },
    {
        "name": "1024_640",
        "w": 1024,
        "h": 640,
    },
    {
        "name": "640_1024",
        "w": 640,
        "h": 1024,
    },
    {
        "name": "1024_424",
        "w": 1024,
        "h": 424,
    },
    {
        "name": "1024_552",
        "w": 1024,
        "h": 552,
    },
    {
        "name": "1024_744",
        "w": 1024,
        "h": 744,
    },
    {
        "name": "1024_720",
        "w": 1024,
        "h": 720,
    },
    {
        "name": "720_1024",
        "w": 720,
        "h": 1024,
    },
]

aspect_ratios = {k["name"]: (k["w"], k["h"]) for k in aspect_ratio_list}
