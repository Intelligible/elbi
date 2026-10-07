Fixed testing a data source connection reporting the driver's error unmasked. A driver
that quoted its connection details when failing would have shown the saved password.
The error still names the host and the cause, now with the password masked.
