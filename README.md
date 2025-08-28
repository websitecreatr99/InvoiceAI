# 📄 Invoice Extractor from Unstructured PDF Data

## ClearDocs (PDF → Structured Data Extractor)

ClearDocs is a lightweight tool that extracts structured data (like headers, line items, and schema) from **unstructured PDFs** using OCR and AI.  
It’s built for **purchase orders, invoices, and other business documents** where data accuracy and structure matter.

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/cleardocs.git
cd cleardocs

2️⃣ Install Dependencies

Make sure you have Python 3.9+ installed. Then run:

pip install -r requirements.txt

3️⃣ Create Environment Variables

Create a .env file in the project root and add your Gemini API Key:

GEMINI_API_KEY=your_api_key_here


🔑 You can get a Gemini API key from Google AI Studio
.

4️⃣ Run the App

Start the application with:

python app.py

🖥️ Usage

Open your browser and go to: http://127.0.0.1:5000/

Upload one or more PDFs

Process them and view extracted JSON schema in the UI

Cancel/remove unwanted files before processing

📂 Project Structure
.
├── app.py                # Main Flask app
├── ocr_pipeline.py       # OCR + extraction pipeline
├── static/               # CSS & JS files
├── templates/            # HTML frontend
├── structured_output/    # Extracted JSON results
├── requirements.txt      # Dependencies
└── .env                  # Your Gemini API key

✅ Features

📂 Upload multiple PDFs

❌ Remove unwanted files before processing

🧾 Extract structured JSON (headers, line items, schema)

🎨 Simple, modern UI

📜 License

MIT License. Free to use and modify.


Do you want me to also include a **preview screenshot section** in the README (with `![screenshot](path/to/img.png)`)