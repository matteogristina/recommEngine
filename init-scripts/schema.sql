--
-- Look-up tables for normalizing string values.
-- These must be created and populated first.
--

CREATE TABLE IF NOT EXISTS product_types (
    product_type_no INTEGER PRIMARY KEY,
    product_type_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS apparel_collections (
    index_group_no INTEGER PRIMARY KEY,
    index_group_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    section_no INTEGER PRIMARY KEY,
    section_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS garment_groups (
    garment_group_no INTEGER PRIMARY KEY,
    garment_group_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS perceived_colors (
    perceived_colour_master_id INTEGER PRIMARY KEY,
    perceived_colour_master_name TEXT NOT NULL
);

--
-- Main products table, keyed by the product_code (the garment style).
--
CREATE TABLE IF NOT EXISTS products (
    product_code BIGINT PRIMARY KEY,
    prod_name TEXT NOT NULL,
    detail_desc TEXT,
    
    -- Foreign keys linking to the lookup tables
    product_type_no INTEGER REFERENCES product_types(product_type_no),
    index_group_no INTEGER REFERENCES apparel_collections(index_group_no),
    section_no INTEGER REFERENCES sections(section_no),
    garment_group_no INTEGER REFERENCES garment_groups(garment_group_no)
);

--
-- Articles table, representing a specific SKU (Stock Keeping Unit).
--
CREATE TABLE IF NOT EXISTS articles (
    article_id BIGINT PRIMARY KEY,
    image_path TEXT,
    
    -- Foreign key linking back to the main products table
    product_code BIGINT REFERENCES products(product_code),
    
    -- Foreign key linking to the colors lookup table
    perceived_colour_master_id INTEGER REFERENCES perceived_colors(perceived_colour_master_id)
);

--
-- Customers table, containing metadata for each customer.
--
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(64) PRIMARY KEY,
    club_member_status TEXT,
    fashion_news_freq TEXT,
    age INTEGER,
    postal_code VARCHAR(64)
);

--
-- Transactions table, representing a specific SKU (Stock Keeping Unit).
--
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    t_dat DATE NOT NULL,
    
    -- This is the crucial column that links the transaction to a user.
    customer_id VARCHAR(64) REFERENCES customers(customer_id) NOT NULL,
    
    -- Foreign key linking back to the main articles table
    article_id BIGINT REFERENCES articles(article_id),
    
    -- Price
    price NUMERIC(10,4) NOT NULL
);