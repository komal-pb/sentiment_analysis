from analyze import analyze_text

if __name__ == "__main__":
    while True:
        text = input("Enter text (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        analyze_text(text)
