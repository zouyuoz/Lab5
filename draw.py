long_text="""

"""

# Split the text into lines
lines = long_text.splitlines()

# Filter out lines that start with "Grad norm: " (ignoring leading/trailing whitespace)
# We also check for the full width space and other characters present in the log.
processed_lines = []
for line in lines:
    stripped_line = line.strip()
    # Check if the line starts with "Grad norm: " after stripping spaces, including the full width space from the input
    if not (stripped_line.startswith("Grad norm:")):
        processed_lines.append(line)

# Join the remaining lines back into a single string
processed_text = "\n".join(processed_lines)

# Save the processed text to a .txt file
output_filename = "processed_log.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(processed_text)

print(f"Processed text saved to {output_filename}")