# CourseLens – Updated Flask Dashboard

## What changed

1. **Dashboard and Course Explorer are separate views**
   - Dashboard contains only KPIs and analytics.
   - Course Explorer is a separate tab in the left navigation.

2. **All dashboard analytics are filter-driven**
   - Organization
   - Difficulty
   - Course Type
   - Duration
   - Minimum Rating
   - KPIs and charts update together.

3. **Review counts are handled without truncation**
   - Existing `review_count` values from the CSV are preserved.
   - Values are displayed with full integer formatting, e.g. `3,246`, never `324`.
   - For rows where the CSV has no review count, the **Fetch missing reviews** button reads the course URL and attempts to extract the full review count.
   - Retrieved values are cached in `review_cache.json`.
   - If Coursera blocks a request or changes its page structure, that individual value remains unavailable rather than being fabricated.

4. **More professional visual language**
   - Font Awesome icons replace the old diamond placeholders.
   - Organizations get recognizable brand icons where supported and a clean building icon fallback.
   - Skills use context-specific icons (code, database, cloud, AI, security, analytics, etc.).
   - Cleaner cards, spacing, filters, tables and modal layout.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Flask address shown in the terminal.

## Review count note

The supplied CSV contains `review_count` for only some rows. The dashboard therefore does not invent missing values. Use **Fetch missing reviews** while connected to the internet to populate values from the course URLs when Coursera allows the request.
