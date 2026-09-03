-- source: warehouse
-- Highest-spending customers from the sales table.
SELECT customer_id, SUM(CAST(amount AS DOUBLE)) AS total
FROM sales
GROUP BY customer_id
ORDER BY total DESC
