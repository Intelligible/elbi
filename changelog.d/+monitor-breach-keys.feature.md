Added a `key` option to monitors over a derivation, naming the rows that breach. Every
alert from such a monitor now lists those rows, and a row that starts breaching while the
monitor is already alerting raises `metric.breach_widened`, emailed by default. Without a
key, a second failing row only moved the total and was never announced.
