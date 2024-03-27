import sqlite3
from io import BytesIO
from PIL import Image
import os


def delete_row_by_email(email_to_delete):
    # Connect to the database
    conn = sqlite3.connect('instance/site.db')
    cursor = conn.cursor()

    try:
        # Execute the DELETE query to remove the row with the specified ID
        cursor.execute('DELETE FROM User WHERE email = ?', (email_to_delete,))
        result = cursor.fetchone()

        # Output the result for debugging
        print(f"Query result for email {email_to_delete}: {result}")

        conn.commit()

        # Delete the local image file
        file_to_delete = f'data/iris_image_{email_to_delete}.png'
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            print(f"Row with EMAIL {email_to_delete} and associated file {file_to_delete} successfully deleted.")
        else:
            print(f"Row with EMAIL {email_to_delete} deleted, but associated file {file_to_delete} not found.")

    except sqlite3.Error as e:
        print(f"Error deleting row: {e}")
    finally:
        # Close the database connection
        conn.close()


def save_image(desired_email):
    # Connect to the database
    conn = sqlite3.connect('instance/site.db')
    cursor = conn.cursor()

    try:
        # Execute the query to retrieve iris_picture associated with the id
        cursor.execute('SELECT iris_picture FROM User WHERE email = ?', (desired_email,))
        result = cursor.fetchone()

        # Close the database connection
        conn.close()

        # Check if an image was found
        if result:
            iris_picture_blob = result[0]

            # Convert the blob into an image object
            iris_picture = Image.open(BytesIO(iris_picture_blob))

            # Save the image in PNG format with a filename based on the id
            file_name = f'data/iris_image_{desired_email}.bmp'
            iris_picture.save(file_name, 'BMP')

            print(f"Image saved successfully as {file_name}")
        else:
            print(f"No image found for the specified email: {desired_email}.")
    except sqlite3.Error as e:
        print(f"Error searching for image: {e}")
    finally:
        # Close the database connection
        conn.close()