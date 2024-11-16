DROP TABLE IF EXISTS vmnc.vmnc_province;

CREATE TABLE vmnc.vmnc_province (
    province VARCHAR(255) PRIMARY KEY,
    total_user INT
);


INSERT INTO vmnc.vmnc_province
SELECT * FROM {{df_1 }};