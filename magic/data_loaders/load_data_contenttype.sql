DROP TABLE IF EXISTS vmnc.vmnc_content_type;

CREATE TABLE vmnc.vmnc_content_type (
    province VARCHAR(255) PRIMARY KEY,
    total_user INT
);


INSERT INTO vmnc.vmnc_content_type
SELECT 
    content_type,
    COUNT(DISTINCT iduser) AS total_user
FROM vmnc.vmnc_clean
GROUP BY 1 
ORDER BY 2 DESC;