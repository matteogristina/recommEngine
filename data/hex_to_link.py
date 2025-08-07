import pandas as pd

def find_link(row):
    translation_table = str.maketrans("", "", "['] ")
    hex_list = row['image_downloads'].translate(translation_table).split(',')
    link_list = row['images'].translate(translation_table).split(',')

    index = hex_list.index(row['main_image'])
    row['main_image'] = link_list[index]
    return row['main_image']


if __name__ == '__main__':
    # Load your CSV file
    csv_file_path = 'store_zara_img_short.csv'
    df = pd.read_csv(csv_file_path)

    # Use the correct column names from your CSV
    links = 'images'
    filenames_col = 'image_downloads'
    main_image_hex = 'main_image'


    df[main_image_hex] = df.apply(find_link, axis=1)
    pd.set_option('display.max_colwidth', None)  # Display full content in columns
    print(df[main_image_hex])






