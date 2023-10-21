import pandas as pd
import re


def convert_html_table_to_color():
    from bs4 import BeautifulSoup
    path = '/home/gao/Documents/temp/latex_color.html'
    data = []
    soup = BeautifulSoup(open(path), 'html.parser')
    header = ["name", "hex", "r", "g", 'b', 'a']
    HTML_data = soup.find_all("table")[0].find_all("tr")[1:]

    regex = re.compile('[^a-zA-Z_-]')

    for element in HTML_data:
        sub_data = []
        for sub_element in element:
            try:
                text = sub_element.get_text().strip()
                if text and "definecolor" not in text:
                    sub_data.append(text)
            except:
                continue
        sub_data[0] = '_'.join(re.split(r'[ -]', sub_data[0])).lower()
        sub_data[0] = regex.sub('', sub_data[0])
        sub_data[2] = float(sub_data[2])
        sub_data[3] = float(sub_data[3])
        sub_data[4] = float(sub_data[4])
        sub_data.append(1.0)
        data.append(sub_data)

    # dataFrame = pd.DataFrame(data=data, columns=header)
    # dataFrame.to_csv('colors.csv')

    with open("./latex_colors_rgba.py", "w") as f:
        for d in data:
            f.write(f"{d[0]} = [{d[2]}, {d[3]}, {d[4]}, {d[5]}]\n")
