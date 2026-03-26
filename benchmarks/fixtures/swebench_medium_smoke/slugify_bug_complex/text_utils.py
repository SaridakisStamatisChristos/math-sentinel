def slugify(value):
    s = " ".join(str(value or "").split())
    return s.strip().lower().replace(" ", "-")
