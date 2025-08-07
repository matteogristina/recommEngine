-- Create the products table only if it does not already exist
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    sku VARCHAR(255) UNIQUE,
    name VARCHAR(255),
    description TEXT,
    price DECIMAL(10, 2),
    terms VARCHAR(255),
    section VARCHAR(255),
    main_image_url TEXT
);

-- Then, create the product_images table only if it does not already exist
-- Note: The `FOREIGN KEY` constraint will only be added if the table is created.
-- This works because the `products` table would have already been created
-- on the first run.
CREATE TABLE IF NOT EXISTS product_images (
    image_id SERIAL PRIMARY KEY,
    product_id INT REFERENCES products(product_id),
    image_url TEXT
);