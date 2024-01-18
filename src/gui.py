import customtkinter

from src.load_data import load_jobs
from src.recommendations import deep_learning, get_recommendations_deep

class UserProfileInputFrame(customtkinter.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master)
        self.master = master
        self.grid_columnconfigure((0, 1), weight=1)
        
        self.title = customtkinter.CTkLabel(self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=3)
        
        self.experience_label = customtkinter.CTkLabel(self, text="Enter your experience in year:", anchor="w")
        self.experience_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.experience_entry = customtkinter.CTkEntry(self, placeholder_text='5')
        self.experience_entry.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="ew")
        
        self.skills_label = customtkinter.CTkLabel(self, text="Enter your skills in this format: skill 1 | skill 2 | ...", anchor="w")
        self.skills_label.grid(row=3, column=0, padx=10, pady=10, sticky="ew", columnspan=2)
        
        self.skills_entry = customtkinter.CTkEntry(self, width=100, placeholder_text='Something like: monitoring | ims | social media | academic instructor | mentor | pop | teaching | training | testing')
        self.skills_entry.grid(row=4, column=0, padx=10, pady=10, sticky="ew", columnspan=2)
        
        self.button = customtkinter.CTkButton(self, text="Search jobs", command=self.button_callback)
        self.button.grid(row=5, column=0, padx=10, pady=10, sticky="ew", columnspan=2)
        
        self.error_label = customtkinter.CTkLabel(self, text='', text_color='#F00')
        
    def button_callback(self):
        experience = self.experience_entry.get()
        skills = self.skills_entry.get().lower()
        self.error_label.grid_forget()
        try:
            experience = int(experience)
            if experience < 0:
                self.error_label.configure(text='Experience must be superior to 0')
                self.error_label.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew", columnspan=2)
            self.master.get_recommendations(skills, experience)
        except ValueError:
            self.error_label.configure(text='Experience must be a positive integer')
            self.error_label.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew", columnspan=2)


class JobsRecommendationFrame(customtkinter.CTkFrame):
    def __init__(self, master, title):
        super().__init__(master)
        self.master = master
        self.grid_columnconfigure(0, weight=1)
        
        self.title = customtkinter.CTkLabel(self, text=title, fg_color="gray30", corner_radius=6)
        self.title.grid(row=0, column=0, padx=10, pady=10, sticky="ew", columnspan=3)
        
        self.textbox = customtkinter.CTkTextbox(master=self, width=400, height=400)
        self.textbox.grid(row=1, column=0, padx=10, sticky="nsew")
        
    def set_text(self, text):
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", text)

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Job Recommendation")
        self.geometry("760x480")
        self.grid_columnconfigure((0,1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        customtkinter.set_appearance_mode("dark")
        
        self.model_created = False

        # User profile inputs
        self.user_profile_input = UserProfileInputFrame(self, 'User profile')
        self.user_profile_input.grid(row=0, column=0, sticky="nsew")
        
        # Jobs recommendation
        self.job_recommendation_frame = JobsRecommendationFrame(self, "Suggested jobs")
        self.job_recommendation_frame.grid(row=0, column=1, sticky="nsew")
        
        # Load data
        self.jobs = load_jobs()
        
    def get_recommendations(self, skills, experience):
        if not self.model_created:
            self.vectorizer, self.embeddings_model, self.knn_model = deep_learning(self.jobs)
            self.model_created = True
        recommendations = get_recommendations_deep(self.jobs, self.knn_model, self.embeddings_model, self.vectorizer, skills, experience)
        self.job_recommendation_frame.set_text(recommendations)
        