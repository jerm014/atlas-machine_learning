import tkinter as tk
from tkinter import messagebox

class CodependencyScaleApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Potter-Efron Codependency Scale")
        self.master.geometry("600x500")

        self.questions = [
            "I tend to take on too much responsibility for others' feelings and problems.",
            "I often put others' needs before my own.",
            "I have difficulty making decisions.",
            "I fear being abandoned or alone.",
            "I have an extreme need for approval and recognition.",
            "I have difficulty identifying my feelings.",
            "I have difficulty expressing my feelings.",
            "I tend to minimize, alter, or deny how I truly feel.",
            "I tend to judge everything I do, think, or say harshly.",
            "I have low self-esteem.",
        ]
        self.questions = {"Fear": [
            "I become preoccupied with the problems of others",
            "I try to “keep things under control” or “keep a handle” on situations",
            "I am afraid to approach others directly",
            ],
        "Shame/Guilt": [
            "I often feel ashamed not only about my behavior but also about the behavior of others",
            "I feel guilty about the problems of others in my family",
            "I sometimes hate myself?",
            ],
        "Prolonged Despair": [
            "I often feel hopeless about changing the current situation",
            "I tend to be pessimistic about the world in general",
            "I have a sense of low self-worth or failure that does not reflect my skills and accomplishments",
            ],
        "Rage": [
            "I feel persistently angry with other family members or with myself",
            "I am afraid of losing control if I you let myself get really mad",
            "I am angry at God / Nature / The Universe",
            ],
        "Denial": [
            "I feel myself denying the basic problems in my family",
            "I tell myself that these problems are not that bad",
            "I find reasons to justify the irresponsible behavior of others in my family",
            ]}

        self.answers = {"Fear": [0,0,0], "Shame/Guilt": [0,0,0], "Prolonged Despair": [0,0,0], "Rage": [0,0,0], "Denial": [0,0,0]}
        self.current_section = "Fear"
        self.current_question = 0

        self.question_label = tk.Label(master, text="", wraplength=500, justify="center")
        self.question_label.pack(pady=20)

        self.scale = tk.Scale(master, from_=-4, to=4, orient="horizontal", length=400, 
                              label="-4 (Never) to 4 (Always)")
        self.scale.pack()

        self.next_button = tk.Button(master, text="Next", command=self.next_question)
        self.next_button.pack(pady=20)

        self.show_question()

    def next_section(self):
        sections = list(self.questions.keys())
        current_index = sections.index(self.current_section)
        if current_index < len(sections) - 1:
            self.current_section = sections[current_index + 1]
        else:
            self.current_section = None
        return self.current_section

    def previous_section(self):
        sections = list(self.questions.keys())
        current_index = sections.index(self.current_section)
        if current_index > 0:
            self.current_section = sections[current_index - 1]
        else:
            self.current_section = None
        return self.current_section


    def show_question(self):
        if self.current_question < len(self.questions):
            self.question_label.config(
              text=f"{self.current_section}: " + \
                self.questions[self.current_section][self.current_question]
            )
            self.scale.set(0)  # Reset to middle value
        else:
            self.show_result()

    def next_question(self):
        self.answers[self.current_section][self.current_question] = self.scale.get()
        self.current_question += 1
        if self.current_question >= 3:
            self.next_section()
            self.current_question = 0
        if self.current_section is None:
            self.show_result()
        else:
            self.show_question()
        #self.show_question()

    def show_result(self):
        total_score = sum(sum(values) for values in self.answers.values())

        result_text = f"Your total score is: {total_score}\n\n"
        if total_score < 20:
            result_text += "You show few signs of codependency."
        elif 20 <= total_score < 30:
            result_text += "You show some signs of codependency. It may be beneficial to explore this further."
        elif 30 <= total_score < 40:
            result_text += "You show moderate signs of codependency. Consider seeking support or counseling."
        else:
            result_text += "You show significant signs of codependency. It's recommended to seek professional help."

        messagebox.showinfo("Result", result_text)
        print("All questions have been answered.")
        for section, values in self.answers.items():
            print(f"{section:<20}: {CodependencyScaleApp.int_to_ascii_scale(sum(values), 18, fill_left='*')} {values}")
        self.master.quit()

    @staticmethod
    def int_to_ascii_scale(
            value,
            total_width=20,
            min_value=-4,
            max_value=4,
            marker="|",
            fill_left=" ",
            fill_right=" "):
        
        if not min_value <= value <= max_value:
            raise ValueError(f"Value must be between {min_value} and {max_value}")
        
        if total_width < 3:  # Minimum width to show []
            raise ValueError("Total width must be at least 3")
        
        # Calculate the position of the marker
        scale_width = total_width - 2  # Subtract 2 for the brackets
        value_range = max_value - min_value
        marker_position = int((value - min_value) / value_range * (scale_width - 1))
        
        # Create the scale
        left_fill = fill_left * marker_position
        right_fill = fill_right * (scale_width - marker_position - 1)
        scale = f"[{left_fill}{marker}{right_fill}]"
        
        return scale

if __name__ == "__main__":
    root = tk.Tk()
    app = CodependencyScaleApp(root)
    root.mainloop()

