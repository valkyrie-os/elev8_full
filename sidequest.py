from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
from collections import defaultdict, Counter
import statistics
from enum import Enum
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import os
from openai import OpenAI

@dataclass
class Message:
    sender: str
    content: Dict[str, Any]
    query_type: str
    timestamp: datetime = field(default_factory=datetime.now)

class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    UNKNOWN = "unknown"

@dataclass
class DataSpecialistAgent:
    conversation_history: List[Message] = field(default_factory=list)
    client: OpenAI = field(default_factory=lambda: OpenAI(api_key="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"))

    def analyze_data(self, data: Dict[str, List[Any]]) -> Dict[str, Any]:
        try:
            print("\nDataSpecialistAgent: Starting data analysis...")
            df = pd.DataFrame(data)
            
            # Analyze column types
            column_types = self._detect_column_types(df)
            
            analysis_results = {
                "column_analysis": {},
                "correlations": {},
                "patterns": {},
                "status": "success"
            }
            
            # Analyze each column based on its type
            for column, dtype in column_types.items():
                print(f"\nAnalyzing column: {column} (Type: {dtype})")
                analysis_results["column_analysis"][column] = self._analyze_column(df[column], dtype)
            
            # Find correlations between numeric columns
            numeric_cols = [col for col, dtype in column_types.items() if dtype == DataType.NUMERIC]
            if len(numeric_cols) > 1:
                analysis_results["correlations"]["numeric"] = self._analyze_numeric_correlations(df[numeric_cols])
            
            # Find patterns between categorical and numeric columns
            cat_cols = [col for col, dtype in column_types.items() if dtype == DataType.CATEGORICAL]
            if numeric_cols and cat_cols:
                analysis_results["patterns"]["cat_numeric"] = self._analyze_categorical_numeric_patterns(
                    df, numeric_cols, cat_cols
                )

            print("Data analysis completed successfully")
            return analysis_results

        except Exception as e:
            print(f"Error in data analysis: {e}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[str, DataType]:
        """Automatically detect the type of each column"""
        column_types = {}
        
        for column in df.columns:
            # Try numeric conversion
            try:
                pd.to_numeric(df[column])
                column_types[column] = DataType.NUMERIC
                continue
            except:
                pass
            
            # Try datetime conversion with explicit format
            try:
                # Try common date formats
                date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                for fmt in date_formats:
                    try:
                        pd.to_datetime(df[column], format=fmt)
                        column_types[column] = DataType.DATETIME
                        break
                    except:
                        continue
                if column not in column_types:
                    raise ValueError
            except:
                pass
            
            # Check if categorical (limited unique values)
            unique_ratio = len(df[column].unique()) / len(df[column])
            if unique_ratio < 0.5:  # If less than 50% unique values, consider categorical
                column_types[column] = DataType.CATEGORICAL
            else:
                # Assume text if strings with high uniqueness
                if df[column].dtype == object:
                    column_types[column] = DataType.TEXT
                else:
                    column_types[column] = DataType.UNKNOWN
        
        return column_types

    def _analyze_column(self, series: pd.Series, dtype: DataType) -> Dict[str, Any]:
        """Analyze a single column based on its type"""
        analysis = {"type": dtype.value}
        
        if dtype == DataType.NUMERIC:
            analysis.update({
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()) if len(series) > 1 else 0,
                "min": float(series.min()),
                "max": float(series.max()),
                "quartiles": series.quantile([0.25, 0.5, 0.75]).to_dict()
            })
        
        elif dtype in [DataType.CATEGORICAL, DataType.TEXT]:
            value_counts = series.value_counts()
            analysis.update({
                "unique_values": len(value_counts),
                "most_common": value_counts.head(5).to_dict(),
                "distribution": (value_counts / len(series)).head(5).to_dict()
            })
        
        elif dtype == DataType.DATETIME:
            analysis.update({
                "min_date": str(series.min()),
                "max_date": str(series.max()),
                "range_days": (series.max() - series.min()).days
            })
        
        return analysis

    def _analyze_numeric_correlations(self, numeric_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        corr_matrix = numeric_df.corr()
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": self._get_strong_correlations(corr_matrix)
        }

    def _analyze_categorical_numeric_patterns(
        self, df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]
    ) -> Dict[str, Any]:
        """Analyze patterns between categorical and numeric columns"""
        patterns = {}
        
        for cat_col in cat_cols:
            patterns[cat_col] = {}
            for num_col in numeric_cols:
                grouped_stats = df.groupby(cat_col)[num_col].agg([
                    'mean', 'median', 'std', 'count'
                ]).to_dict('index')
                patterns[cat_col][num_col] = grouped_stats
        
        return patterns

    def _get_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """Identify strong correlations above a threshold"""
        strong_corr = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    strong_corr[f"{col1}_vs_{col2}"] = float(corr)
        return strong_corr

    def _analyze_outliers(self, data: Dict[str, List[Any]], numeric_columns: List[str]) -> Dict[str, Any]:
        """Analyze outliers in numeric columns using IQR method"""
        try:
            outlier_analysis = {}
            
            for column in numeric_columns:
                values = pd.Series(data[column])
                
                # Calculate IQR and bounds
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Find outliers
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                
                # Calculate impact on mean
                if not outliers.empty:
                    mean_with_outliers = values.mean()
                    mean_without_outliers = values[~values.isin(outliers)].mean()
                    impact = abs((mean_with_outliers - mean_without_outliers) / mean_with_outliers * 100)
                else:
                    impact = 0
                
                outlier_analysis[column] = (
                    outliers.to_dict() if not outliers.empty else "No outliers detected",
                    impact
                )
            
            return outlier_analysis
            
        except Exception as e:
            print(f"Error in outlier analysis: {e}")
            return {}

    def _analyze_relationships(self, data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze relationships between different columns"""
        try:
            df = pd.DataFrame(data)
            relationships = {
                "correlations": {},
                "patterns": {}
            }
            
            # Analyze numeric correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                relationships["correlations"] = {
                    "matrix": corr_matrix.to_dict(),
                    "strong_pairs": self._find_strong_correlations(corr_matrix)
                }
            
            # Analyze categorical relationships
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                for cat_col in cat_cols:
                    relationships["patterns"][cat_col] = {}
                    for num_col in numeric_cols:
                        grouped_stats = df.groupby(cat_col)[num_col].agg([
                            'mean', 'median', 'std', 'count'
                        ]).to_dict('index')
                        relationships["patterns"][cat_col][num_col] = grouped_stats
            
            return relationships
            
        except Exception as e:
            print(f"Error in relationship analysis: {e}")
            return {}

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """Find strong correlations above threshold"""
        strong_pairs = {}
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    strong_pairs[f"{col1}_vs_{col2}"] = float(corr)
                    
        return strong_pairs

    def handle_clarification(self, query: Message, raw_data: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Handle clarification requests from the report generator"""
        try:
            print(f"\nDataSpecialist: Processing clarification request - {query.content}")
            
            if query.query_type == "clarification":
                if query.content.get("query") == "distribution_details":
                    return self._analyze_distribution_details(
                        raw_data, 
                        query.content.get("columns", [])
                    )
                # Add more query type handlers as needed
            
            return {
                "status": "error",
                "message": "Unknown clarification request type"
            }
            
        except Exception as e:
            print(f"Error handling clarification: {e}")
            return {
                "status": "error",
                "message": f"Clarification failed: {str(e)}"
            }

    def _analyze_distribution_details(self, data: Dict[str, List[Any]], columns: List[str]) -> Dict[str, Any]:
        """Analyze detailed distribution for specified columns"""
        try:
            details = {}
            df = pd.DataFrame(data)
            
            for column in columns:
                if column in df.columns:
                    series = df[column]
                    details[column] = {
                        "quartiles": series.quantile([0.25, 0.5, 0.75]).to_dict(),
                        "skewness": float(series.skew()),
                        "kurtosis": float(series.kurtosis()),
                        "histogram_data": np.histogram(series, bins='auto')
                    }
            
            return {
                "status": "success",
                "distribution_details": details
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Distribution analysis failed: {str(e)}"
            }

    def analyze_with_ai(self, data: Dict[str, List[Any]]) -> Dict[str, Any]:
        try:
            # Convert data to a readable format for GPT
            data_summary = self._prepare_data_summary(data)
            
            prompt = f"""
            As a data analyst, analyze this business data and provide insights:
            {data_summary}
            
            Please provide:
            1. Key Performance Insights
            2. Concerning Trends
            3. Recommendations
            4. Notable Correlations
            5. Customer Feedback Analysis
            
            Focus on actionable business insights.
            """

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{
                    "role": "system",
                    "content": "You are an expert data analyst focused on retail business performance."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7
            )

            return {
                "ai_analysis": response.choices[0].message.content,
                "status": "success"
            }

        except Exception as e:
            print(f"AI Analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _prepare_data_summary(self, data: Dict[str, List[Any]]) -> str:
        """Format data for GPT analysis"""
        summary = []
        for key, values in data.items():
            if isinstance(values[0], (int, float)):
                summary.append(f"{key}: avg={statistics.mean(values):.2f}, range={min(values)}-{max(values)}")
            else:
                unique_vals = Counter(values)
                summary.append(f"{key}: {dict(unique_vals)}")
        return "\n".join(summary)

    def ask_ai(self, question: str) -> Dict[str, Any]:
        try:
            print(f"\nDataSpecialist: Processing question - {question[:100]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert business analyst. Provide specific, data-driven insights."
                    },
                    *[{"role": "assistant" if msg.sender == "AI" else "user",
                       "content": str(msg.content)} 
                      for msg in self.conversation_history[-5:]],  # Include last 5 messages for context
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0.7
            )

            # Store the interaction
            self.conversation_history.append(Message(
                sender="user",
                content=question,
                query_type="analysis"
            ))
            self.conversation_history.append(Message(
                sender="AI",
                content=response.choices[0].message.content,
                query_type="response"
            ))

            return {
                "ai_analysis": response.choices[0].message.content,
                "status": "success"
            }

        except Exception as e:
            print(f"AI Query failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "ai_analysis": "Analysis failed due to an error."
            }

@dataclass
class ReportGeneratorAgent:
    def _format_statistical_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format the statistical analysis results into a readable string"""
        formatted = []
        
        # Format column analysis
        if "column_analysis" in analysis:
            formatted.append("Column Analysis:")
            for col, details in analysis["column_analysis"].items():
                formatted.append(f"\n{col}:")
                for key, value in details.items():
                    formatted.append(f"  {key}: {value}")
        
        # Format correlations
        if "correlations" in analysis and "numeric" in analysis["correlations"]:
            formatted.append("\nStrong Correlations:")
            corr_data = analysis["correlations"]["numeric"]
            if "strong_correlations" in corr_data:
                for pair, value in corr_data["strong_correlations"].items():
                    formatted.append(f"  {pair}: {value:.3f}")
        
        # Format patterns
        if "patterns" in analysis and "cat_numeric" in analysis["patterns"]:
            formatted.append("\nCategory-Numeric Patterns:")
            for cat, metrics in analysis["patterns"]["cat_numeric"].items():
                formatted.append(f"\n{cat}:")
                for metric, values in metrics.items():
                    formatted.append(f"  {metric}:")
                    for category, stats in values.items():
                        formatted.append(f"    {category}: {stats}")
        
        return "\n".join(formatted)

    def generate_report(self, analysis: Dict[str, Any], data_specialist: DataSpecialistAgent, raw_data: Dict[str, List[Any]]) -> str:
        try:
            report = "\nCOMPREHENSIVE BUSINESS ANALYSIS\n" + "="*40 + "\n"

            # 1. Initial Analysis
            print("\nReportGenerator: Requesting initial overview...")
            initial = data_specialist.ask_ai(
                "Provide an initial overview of the business performance based on this data."
            )
            report += "\n1. INITIAL OVERVIEW\n" + "-"*20 + "\n"
            report += initial["ai_analysis"] + "\n"

            # 2. Deep dive into concerning metrics
            print("\nReportGenerator: Investigating concerning metrics...")
            concerning = data_specialist.ask_ai(
                f"Looking at these correlations: {analysis['correlations']}, what are the most concerning relationships and why?"
            )
            report += "\n2. CONCERNING METRICS ANALYSIS\n" + "-"*20 + "\n"
            report += concerning["ai_analysis"] + "\n"

            # 3. Customer Experience Analysis
            print("\nReportGenerator: Analyzing customer experience...")
            customer = data_specialist.ask_ai(
                f"Analyze the customer experience through: customer_feedback, ratings decline (morning: {raw_data.get('morning_rating', [])}, evening: {raw_data.get('evening_rating', [])}), and customer count variations."
            )
            report += "\n3. CUSTOMER EXPERIENCE INSIGHTS\n" + "-"*20 + "\n"
            report += customer["ai_analysis"] + "\n"

            # 4. Store Performance Analysis
            print("\nReportGenerator: Analyzing store performance patterns...")
            store = data_specialist.ask_ai(
                f"Compare performance between store types (mall vs street) and locations (urban/suburban/rural). Consider revenue, customer count, and satisfaction metrics."
            )
            report += "\n4. STORE PERFORMANCE PATTERNS\n" + "-"*20 + "\n"
            report += store["ai_analysis"] + "\n"

            # 5. Final Recommendations
            print("\nReportGenerator: Requesting final recommendations...")
            recommendations = data_specialist.ask_ai(
                "Based on all previous analysis, what are the top 3 most critical actions this business should take?"
            )
            report += "\n5. CRITICAL RECOMMENDATIONS\n" + "-"*20 + "\n"
            report += recommendations["ai_analysis"] + "\n"

            # Add statistical analysis at the end
            report += "\nSTATISTICAL DETAILS\n" + "="*40 + "\n"
            report += self._format_statistical_analysis(analysis)

            return report

        except Exception as e:
            return f"Error generating report: {str(e)}"

def run_analysis(data: Dict[str, List[Any]]) -> str:
    """Run analysis on any dataset"""
    print("\nStarting flexible data analysis...")
    
    try:
        data_specialist = DataSpecialistAgent()
        report_generator = ReportGeneratorAgent()
        
        analysis_results = data_specialist.analyze_data(data)
        final_report = report_generator.generate_report(
            analysis_results,
            data_specialist,
            data
        )
        
        return final_report
        
    except Exception as e:
        return f"Analysis failed: {str(e)}\n{traceback.format_exc()}"

if __name__ == "__main__":
    # Test data
    sample_data = {
        "sales": [120, 150, 80, 200, 90],
        "customer_feedback": ["great", "poor", "great", "medium", "poor"],
        "region": ["north", "south", "north", "east", "west"]
    }
    
    result = run_analysis(sample_data)
    print(result)
