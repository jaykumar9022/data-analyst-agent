# data-analyst-agent
A data analyst agent exposed as a FastAPI API endpoint. It leverages Large Language Models to process natural language data analysis tasks, including web scraping, data querying, and generating visualizations.

your-project-folder/
 ├── main.py             # Your FastAPI application code
 ├── requirements.txt    # List of all Python dependencies
 ├── .env                # Your API keys and sensitive environment variables (DO NOT COMMIT!)
 ├── .gitignore          # Specifies files/folders to ignore in Git
 ├── Dockerfile          # (Optional but recommended) For containerized deployment
 ├── LICENSE             # The MIT License file

**What to Do on GitHub**
*Your journey starts here. GitHub will host your project's code.*
1. # Create a New Public Repository:
   + Go to github.com/new and log in.
   + Repository name: Give it a clear name, like data-analyst-agent.
   + Description (Optional): Add a short phrase, e.g., "An AI agent that analyzes and visualizes data."
   + Public: Make sure "Public" is selected. This is crucial for evaluation.
   + Add a README file: Check this box.
   + Add .gitignore: Choose the "Python" template from the dropdown. This automatically ignores unnecessary files.
   + Choose a license: Select "MIT License" from the dropdown. This is a project requirement.
   + Click the green "Create repository" button.

2. # Clone Your Repository to Your Computer Through Vs Code:
   + Open VS Code
   + Sign in through Github Account then click on **Clone Repository**
   + Then Select your Project 2 Repository
   + A new window pop up for Repository Destination then select **Desktop** then click on Select as Repository Destination
   + Now Your Repository Connect with VS Code
  
3. # Create Files For Project 2
   + main.py
   + requirements.txt
   + .env ()
   + Dockerfile

4. # After Creating these Files now time to upload using commands in Vs code terminal.

5. # Install VS Code Extensions
   + Python
   + Pylance
   + Docker
  
6. # Now Type Below Code in your terminal
   + macOS/Linux: source venv/bin/activate
   + Windows: .\venv\Scripts\activate

7. # Type this => pip install -r requirements.txt

8. # When all requiremnets is completed then type this => uvicorn main:app --reload --host 0.0.0.0 --port 8000

9. # Open your web browser and go to:
      http://localhost:8000/ and http://localhost:8000/docs

10. # In this site http://localhost:8000/docs click to expand POST/api/ , then Click on TRY IT OUT

11. # Now Copy the Below question and Paste the question in *Edit Value|schema* Box
    {
  "task": "Scrape the list of highest grossing films from Wikipedia. It is at the URL:\nhttps://en.wikipedia.org/wiki/List_of_highest-grossing_films\n\nAnswer the following questions and respond with a JSON array of strings containing the answer.1. How many $2 bn movies were released before 2020?2. Which is the earliest film that grossed over $1.5 bn?3. What's the correlation between the Rank and Peak?4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.\n   Return as a base-64 encoded data URI, \"data:image/png;base64,iVBORw0KG...\" under 100,000 bytes."
}

# When above question sucessfully completed then try Next question
   {
  "task": "The Indian high court judgement dataset contains judgements from the Indian High Courts, downloaded from [ecourts website](https://judgments.ecourts.gov.in/). It contains judgments of 25 high courts, along with raw metadata (as .json) and structured metadata (as .parquet). ... [paste the rest of your Indian High Court question.txt content here, ensuring it's a single string for the 'task' key]"
}


12. # If both questions give correct answer then you deploy your Projetc..
    
   
