import cohere
from typing import Dict
from git import Repo
from datetime import datetime
import os
from typing import List, Dict
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class CommitAnalyzerLLM:
    def __init__(self, api_key: str):
        """Initialize with Cohere API key"""
        self.client = cohere.Client(api_key)

    def analyze_commit(self, commit_data: Dict) -> Dict:
        """Analyze a single commit using Cohere"""
        prompt = self._create_analysis_prompt(commit_data)
        
        try:
            response = self.client.chat(
                model="command-r-plus",  # Using Cohere's command model
                message=prompt,
                temperature=0.3,
                chat_history=[
                    {"role": "System", "message": "You are a code review expert. Analyze the following commit and provide a rating and explanation."}
                ]
            )
            
            return {
                'commit_hash': commit_data['hash'],
                'analysis': response.text,
                'rating': self._extract_rating(response.text)
            }
        except Exception as e:
            return {
                'commit_hash': commit_data['hash'],
                'analysis': f"Error analyzing commit: {str(e)}",
                'rating': 0
            }

    def _create_analysis_prompt(self, commit_data: Dict) -> str:
        """Create a prompt for the LLM based on commit data"""
        return f"""
        Analyze the following commit:
        
        Commit Hash: {commit_data['hash']}
        Author: {commit_data['author']}
        Date: {commit_data['date']}
        Message: {commit_data['message']}
        
        Changes:
        - Files changed: {', '.join(commit_data['changes']['files_changed'])}
        - Insertions: {commit_data['changes']['insertions']}
        - Deletions: {commit_data['changes']['deletions']}
        
        Please provide:
        1. A rating from 1-10 for commit quality
        2. A brief explanation of the rating
        3. Suggestions for improvement (if any)
        """

    def _extract_rating(self, analysis: str) -> int:
        """Extract numerical rating from analysis text"""
        try:
            for line in analysis.split('\n'):
                if 'rating' in line.lower() and ':' in line:
                    rating_str = line.split(':')[1].strip()
                    return int(rating_str.split('/')[0])
        except:
            return 0
        return 0 
    

    ###################################################################################################

class CommitAnalyzer:
    def __init__(self, repo_path: str):
        """Initialize with either a local repo path or a GitHub URL"""
        self.repo_path = repo_path
        self.repo = self._initialize_repo()

    def _initialize_repo(self) -> Repo:
        """Initialize repository - clone if URL, open if local"""
        if self.repo_path.startswith(('http://', 'https://')):
            # Extract repo name from URL
            repo_name = self.repo_path.split('/')[-1].replace('.git', '')
            if not os.path.exists(repo_name):
                return Repo.clone_from(self.repo_path, repo_name)
            return Repo(repo_name)
        return Repo(self.repo_path)

    def get_commit_history(self) -> List[Dict]:
        """Get all commits with their details"""
        commits_data = []
        
        for commit in self.repo.iter_commits():
            commit_info = {
                'hash': commit.hexsha,
                'author': f"{commit.author.name} <{commit.author.email}>",
                'date': datetime.fromtimestamp(commit.committed_date),
                'message': commit.message.strip(),
                'changes': self._get_commit_changes(commit)
            }
            commits_data.append(commit_info)
            
        return commits_data

    def _get_commit_changes(self, commit) -> Dict:
        """Get detailed changes for a specific commit"""
        if not commit.parents:  # First commit
            return self._get_initial_commit_changes(commit)
        
        parent = commit.parents[0]
        diffs = parent.diff(commit)
        
        changes = {
            'files_changed': [],
            'insertions': 0,
            'deletions': 0
        }
        
        for diff in diffs:
            changes['files_changed'].append(diff.a_path)
            if diff.a_blob and diff.b_blob:
                changes['insertions'] += len(diff.b_blob.data_stream.read().decode('utf-8', errors='ignore').splitlines())
                changes['deletions'] += len(diff.a_blob.data_stream.read().decode('utf-8', errors='ignore').splitlines())
                
        return changes

    def _get_initial_commit_changes(self, commit) -> Dict:
        """Handle the initial commit differently"""
        changes = {
            'files_changed': [],
            'insertions': 0,
            'deletions': 0
        }
        
        # Get the tree of the commit
        tree = commit.tree
        
        # Add all files in the initial commit
        for blob in tree.traverse():
            if blob.type == 'blob':  # If it's a file
                changes['files_changed'].append(blob.path)
                changes['insertions'] += len(blob.data_stream.read().decode('utf-8', errors='ignore').splitlines())
                
        return changes 
    
    ################################################################################################################

def analyze_repository(repo_path: str, cohere_api_key: str) -> List[Dict]:
    """Analyze all commits in a repository"""
    # Initialize analyzers
    commit_analyzer = CommitAnalyzer(repo_path)
    llm_analyzer = CommitAnalyzerLLM(cohere_api_key)
    
    # Get commit history
    commits = commit_analyzer.get_commit_history()
    
    # Analyze each commit
    analyzed_commits = []
    for commit in commits:
        analysis = llm_analyzer.analyze_commit(commit)
        analyzed_commits.append({**commit, 'analysis': analysis})
    
    return analyzed_commits

def save_analysis(analyzed_commits: List[Dict], output_file: str):
    """Save analysis results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(analyzed_commits, f, default=str, indent=2)

class RepositoryRequest(BaseModel):
    repo_url: str
    cohere_api_key: str
    output_file: str = "commit_analysis.json"

app = FastAPI()

@app.get("/analyze-repository")
async def analyze_repo_endpoint():
    """API endpoint to analyze a repository"""
    try:
        # Run analysis
        github_repo_url = "https://github.com/mertincesu/cohere_hackathon_data"
        cohere_api_key = "VkRfL2rl1Swi3YJF8W52J6Rxz4ZJDC0aFoQk68r6"
        analyzed_commits = analyze_repository(github_repo_url, cohere_api_key)
        
        return {
            "status": "success",
            "data": analyzed_commits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Replace the __main__ block with:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 