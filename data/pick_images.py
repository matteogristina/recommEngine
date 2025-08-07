import pandas as pd
import ast
import os
import tkinter as tk
from PIL import Image, ImageTk


class ImageSelectorApp:
    def __init__(self, root, df, item_id_col, filenames_col, selected_image_col):
        self.root = root
        self.df = df
        self.item_id_col = item_id_col
        self.filenames_col = filenames_col
        self.selected_image_col = selected_image_col

        # This path must be correct for your setup.
        # It's relative to where your script is being run.
        self.image_directory = 'images/'

        self.current_item_index = 0
        self.photo_images = []

        self.root.title("Image Selector")

        # UI components
        self.item_label = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.item_label.pack(pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

        self.load_next_item()

    def load_next_item(self):
        # Clear the old images
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        self.photo_images.clear()

        # Find the next item that hasn't been selected
        while self.current_item_index < len(self.df):
            row = self.df.iloc[self.current_item_index]

            # Check for a null/empty value in the selected_image_col
            # The 'is not' check is a safer way to check for empty strings or NaN
            if pd.isna(row[self.selected_image_col]) or row[self.selected_image_col] == '':
                item_id = row[self.item_id_col]

                # Check if the list is empty or contains non-string values
                filenames = row[self.filenames_col]
                if isinstance(filenames, list) and filenames:
                    self.display_images(item_id, filenames)
                    return  # Exit the function after displaying an item
                else:
                    print(f"No valid images found for Item ID {item_id}.")

            self.current_item_index += 1

        # If the loop finishes, all items are done
        self.item_label.config(text="Manual curation complete!")
        print("Manual curation complete!")
        # Add an exit button for the user
        tk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=10)

    def display_images(self, item_id, filenames):
        self.item_label.config(text=f"Select Main Image for SKU: {item_id}")

        for i, filename in enumerate(filenames):
            # Assumes the files are stored as hexadecimal.jpg
            image_path = os.path.join(self.image_directory, filename + ".jpg")

            try:
                original_image = Image.open(image_path)
                thumbnail_size = (250, 250)
                original_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(original_image)
                self.photo_images.append(photo)

                button = tk.Button(self.image_frame, image=photo, command=lambda f=filename: self.on_image_click(f))
                button.grid(row=0, column=i, padx=5, pady=5)

            except FileNotFoundError:
                print(f"Image file not found for {filename} at path {image_path}")

    def on_image_click(self, selected_filename):
        row = self.df.iloc[self.current_item_index]
        item_id = row[self.item_id_col]

        print(f"SKU {item_id}: Selected image {selected_filename}")

        # Update the DataFrame
        # We need to use .loc with a specific index to avoid SettingWithCopyWarning
        self.df.at[self.current_item_index, self.selected_image_col] = selected_filename

        # Save the updated DataFrame after each selection
        self.df.to_csv('store_zara_img_select.csv', index=False)

        # Move to the next item
        self.current_item_index += 1
        self.load_next_item()


if __name__ == '__main__':
    # Load your CSV file
    csv_file_path = 'store_zara_img_select.csv'
    df = pd.read_csv(csv_file_path)

    # Use the correct column names from your CSV
    item_id_col = 'sku'
    filenames_col = 'image_downloads'
    selected_image_col = 'main_image'

    # Ensure the 'main_image' column exists; if not, create it with empty strings
    if selected_image_col not in df.columns:
        df[selected_image_col] = ''

    # Check if the filenames column needs to be parsed (it will be a string initially)
    # This also handles the case where the list might be empty
    df[filenames_col] = df[filenames_col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Start the Tkinter application
    root = tk.Tk()
    app = ImageSelectorApp(root, df, item_id_col, filenames_col, selected_image_col)
    root.mainloop()