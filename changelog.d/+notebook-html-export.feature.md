Notebooks can be downloaded as a shareable page. **⋯ → Download .html** on a notebook, or
`GET /api/notebooks/{id}/export?format=html`, writes one self-contained, read-only HTML file:
markdown rendered, code, and outputs. Outputs follow the `.ipynb` export's rule and are
included only where the deployment sets `NOTEBOOK_EXPORT_OUTPUTS=1`; otherwise the page is
code only and says so. The page allows no script, so kernel HTML in an output stays inert.
