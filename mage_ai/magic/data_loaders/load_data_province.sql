DROP TABLE IF EXISTS vmnc.vmnc_province;

CREATE TABLE vmnc.vmnc_province (
    province VARCHAR(255) PRIMARY KEY,
    total_user INT
);


INSERT INTO vmnc.vmnc_province
SELECT 
    province,
    COUNT(DISTINCT iduser) AS total_user
FROM vmnc.vmnc_clean
GROUP BY 1 
ORDER BY 2 DESC;