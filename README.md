# AI Candidate Evaluation System

An automated candidate evaluation system that processes resumes and interview transcripts using AI models (OpenAI GPT and Google Gemini) to provide structured assessments and hiring recommendations.

## 🚀 Features

- **Resume Analysis**: Automated scoring of education, skills, work experience, and background
- **Interview Evaluation**: Assessment of technical knowledge and role-specific skills from transcripts
- **AI-Powered Summarization**: Comprehensive candidate summaries with hiring recommendations
- **Parallel Processing**: Efficient batch processing with configurable worker threads
- **Multi-Model Support**: Choose between OpenAI GPT and Google Gemini APIs
- **Excel Integration**: Direct input/output with Excel files for easy data management

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API account (for GPT models)
- Google Cloud account with Gemini API access
- Excel file with candidate data

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd evalgv
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root directory:
   ```env
   # OpenAI API Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Google Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   **How to get API keys:**
   - **OpenAI API Key**: 
     1. Go to [OpenAI Platform](https://platform.openai.com/)
     2. Sign up/login and navigate to API Keys
     3. Create a new API key
   
   - **Gemini API Key**: 
     1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
     2. Sign in with Google account
     3. Create a new API key

## 📁 File Structure

```
evalgv/
├── gem.py              # Main script for Google Gemini API
├── open.py             # Main script for OpenAI GPT API
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── Test.xlsx          # Input Excel file (your data)
```

## 🔧 Usage

### Option 1: Using Google Gemini API

```bash
python gem.py
```

**Features:**
- Uses Google Gemini models (gemini-2.5-flash, gemini-2.5-pro)
- Optimized batch processing
- Built-in error handling and retry logic
- Output: `Test_results.xlsx`

### Option 2: Using OpenAI GPT API

```bash
python open.py
```

**Features:**
- Uses OpenAI GPT models (gpt-4o)
- Parallel processing with ThreadPoolExecutor
- Configurable worker threads
- Output: `Toast_results.xlsx`

## 📊 Input Data Format

Your Excel file should contain these columns:

| Column Name | Description |
|-------------|-------------|
| `Grapevine Job - Job → Description` | Job description/role details |
| `Grapevine Aiinterviewinstance → Transcript → Conversation` | Interview transcript |
| `Grapevine Userresume - Resume → Metadata → Resume Text` | Resume content |
| `Recruiter GPT Response` | Job-specific criteria |

## 📈 Output Format

The system generates results with these columns:

| Column Name | Description |
|-------------|-------------|
| `Interview Evaluator Agent (RAG-LLM)` | Interview assessment scores |
| `Resume Evaluator Agent (RAG-LLM)` | Resume evaluation scores |
| `Resume + Interview Summarizer Agent` | AI-generated summary |
| `Result` | Final recommendation (Advanced/Reject/Manual Intervention) |

## ⚙️ Configuration

### Performance Tuning

**gem.py (Gemini)**:
- Modify `max_workers` in the script (default: 8)
- Adjust batch processing parameters
- Configure retry logic and delays

**open.py (OpenAI)**:
- Change `max_workers` variable (default: 8)
- Modify `max_tokens` for response length
- Adjust temperature for creativity (default: 0.2)

### Model Selection

**Gemini Models**:
- `gemini-2.5-flash`: Fast, cost-effective
- `gemini-2.5-pro`: Higher quality, more expensive

**OpenAI Models**:
- `gpt-4o`: Latest, most capable model
- Configurable in `call_openai()` function

## 🔒 Security & Privacy

- ✅ API keys stored in `.env` file (not committed to git)
- ✅ Input data files excluded from version control
- ✅ Built-in anonymization functions
- ✅ Secure error handling for API failures

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: Invalid API key
   ```
   - Check your `.env` file
   - Verify API key format and permissions

2. **File Not Found**
   ```
   Error: Excel file not found
   ```
   - Ensure your input file is named correctly
   - Check file path and permissions

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   - Reduce `max_workers` value
   - Add delays between requests

4. **Library Import Errors**
   ```
   Error: "configure" is not exported from module
   ```
   - Update the Google Generative AI library:
     ```bash
     pip install google-generativeai --upgrade
     ```

## 💡 Tips for Best Performance

1. **Start Small**: Test with a few rows first
2. **Monitor Usage**: Check API quotas and billing
3. **Batch Processing**: Use `gem.py` for large datasets
4. **Data Quality**: Ensure clean, properly formatted input data
5. **Backup Results**: Save outputs before making changes

## 📝 License

This project is for internal use and evaluation purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review error logs
3. Verify API key configuration
4. Test with minimal data first

---

**Note**: This system processes sensitive candidate data. Ensure compliance with data privacy regulations and obtain proper consent before use.
