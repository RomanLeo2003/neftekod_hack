# neftekod_hack

Вот обновлённый `README.md` с **инструкцией по запуску Jupyter Notebook в VS Code**, а также описанием подключения виртуального окружения:

---

### 📘 README.md — Установка окружения и запуск Jupyter Notebook

#### 🐍 Шаг 1. Создайте виртуальное окружение

> Используется `venv`, но можно заменить на `conda`, если нужно.

```bash
python3 -m venv venv
```

#### 📦 Шаг 2. Активируйте окружение

- **Linux / macOS:**

```bash
source venv/bin/activate
```

- **Windows (cmd):**

```cmd
venv\Scripts\activate
```

- **Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

#### 📥 Шаг 3. Установите зависимости из `requirements.txt`

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Убедитесь, что файл `requirements.txt` находится в корне проекта.

#### 📓 Шаг 4. Установите Jupyter и зарегистрируйте ядро

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

> После этого ядро `Python (venv)` будет доступно в интерфейсе Jupyter и VS Code.

---

### 🚀 Шаг 5. Запуск в Jupyter Notebook

```bash
jupyter notebook
```

После запуска Jupyter откроется в браузере или будет показана ссылка вида:

```
http://localhost:8888/tree
```

---

### 🧠 Шаг 6. Как запустить в **Visual Studio Code**

1. Убедитесь, что у вас установлены расширения:
   - **Python**
   - **Jupyter**

2. Откройте папку проекта в VS Code:
   ```bash
   code .
   ```

3. В правом верхнем углу любого `.ipynb`-файла появится выпадающий список для выбора ядра — выберите:
   ```
   Python (venv)
   ```

   Если ядро не отображается, перезапустите VS Code.
