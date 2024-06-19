import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext, ttk
import threading
import os
from Scraper import scrape_website  # Predpostavka, da ta funkcija obstaja

class ScraperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Web Scraper GUI")
        self.geometry("600x400")
        self.download_path = "C:\\Users\\jasar\\Desktop\\TZ_projekt\\prenosi_izvrzbe"  # Mapa za prenose
        self.init_ui()

    def init_ui(self):
        tk.Label(self, text="Datum (dd.mm.yyyy):").pack(pady=(10,0))
        self.datum_entry = tk.Entry(self)
        self.datum_entry.pack(pady=(0,5))

        tk.Label(self, text="Max Frm Pos:").pack(pady=(5,0))
        self.max_frm_pos_entry = tk.Entry(self)
        self.max_frm_pos_entry.pack(pady=(0,5))

        tk.Button(self, text="Start Scraping", command=self.start_scraping_thread).pack(pady=(10,5))
        self.pdf_listbox = tk.Listbox(self, height=15)
        self.pdf_listbox.pack(pady=(5,10), fill=tk.BOTH, expand=True)
        self.pdf_listbox.bind("<Double-1>", self.open_pdf)

        self.refresh_pdf_list()  # Osveži seznam PDF-jev ob zagonu

    def start_scraping_thread(self):
        datum = self.datum_entry.get()
        try:
            max_frm_pos = int(self.max_frm_pos_entry.get())
        except ValueError:
            messagebox.showerror("Napaka", "Max Frm Pos mora biti številka")
            return
        threading.Thread(target=self.scrape, args=(max_frm_pos, datum), daemon=True).start()

    def scrape(self, max_frm_pos, datum):
        scrape_website(max_frm_pos, datum)  # Predvidevamo, da ta funkcija izvede potrebne akcije
        self.after(100, self.refresh_pdf_list)  # Osveži seznam PDF-jev po skriptiranju

    def refresh_pdf_list(self):
        self.pdf_listbox.delete(0, tk.END)  # Počisti trenutni seznam
        for file in os.listdir(self.download_path):
            if file.endswith(".pdf"):
                self.pdf_listbox.insert(tk.END, file)

    def open_pdf(self, event=None):
        selected_index = self.pdf_listbox.curselection()
        if selected_index:
            selected_file = self.pdf_listbox.get(selected_index[0])
            file_path = os.path.join(self.download_path, selected_file)
            os.startfile(file_path)  # Odpre izbrani PDF

if __name__ == "__main__":
    app = ScraperGUI()
    app.mainloop()
