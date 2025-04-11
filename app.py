from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import os
import pandas as pd
import bcrypt
from werkzeug.utils import secure_filename
from train_dynamic import anonymize_data  # Giả sử bạn đã có hàm này
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ANONYMIZED_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANONYMIZED_FOLDER'], exist_ok=True)

# Đường dẫn đến file users.csv
USERS_FILE = 'data/users.csv'

# Đảm bảo thư mục data tồn tại
os.makedirs('data', exist_ok=True)

# Khởi tạo file users.csv nếu chưa tồn tại
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "hashed_password", "role", "theme"]).to_csv(USERS_FILE, index=False)

# Hàm đọc dữ liệu người dùng từ file users.csv
def load_users():
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE)
        users = {}
        for _, row in df.iterrows():
            users[row['username']] = {
                'password': row['hashed_password'].encode('utf-8'),  # Mật khẩu đã mã hóa
                'role': row['role'],
                'theme': row['theme'],
                'history': []  # Lịch sử ẩn danh được lưu tạm trong session
            }
        return users
    return {}

# Hàm lưu dữ liệu người dùng vào file users.csv
def save_users(users):
    data = []
    for username, info in users.items():
        data.append({
            'username': username,
            'hashed_password': info['password'].decode('utf-8') if isinstance(info['password'], bytes) else info['password'],
            'role': info['role'],
            'theme': info['theme']
        })
    df = pd.DataFrame(data)
    df.to_csv(USERS_FILE, index=False)

# Tải dữ liệu người dùng
users = load_users()

# Định nghĩa bộ lọc basename cho Jinja2
@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path) if path else ''

# Route cho trang chính sách sử dụng
@app.route('/policy', methods=['GET', 'POST'])
def policy():
    if request.method == 'POST':
        if 'agree' in request.form:
            session['policy_agreed'] = True
            flash('Bạn đã đồng ý với chính sách sử dụng.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Bạn đã từ chối chính sách sử dụng.', 'danger')
            return redirect(url_for('login'))
    return render_template('policy.html', theme=session.get('theme', 'light'))

# Route cho trang đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Tên người dùng đã tồn tại!', 'danger')
            return redirect(url_for('register'))
        # Mã hóa mật khẩu
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users[username] = {
            'password': hashed_password,
            'role': 'user',
            'theme': 'light',
            'history': []
        }
        save_users(users)  # Lưu vào file users.csv
        flash('Đăng ký thành công! Vui lòng đăng nhập.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', theme=session.get('theme', 'light'))

# Route cho trang đăng nhập
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Trường hợp đăng nhập với tư cách khách
        if 'guest_login' in request.form:
            session['role'] = 'guest'
            session['theme'] = 'light'  # Giao diện mặc định cho khách
            flash('Bạn đã đăng nhập với tư cách khách!', 'success')
            return redirect(url_for('policy'))

        # Trường hợp đăng nhập thông thường
        username = request.form.get('username')
        password = request.form.get('password')

        # Kiểm tra nếu username hoặc password trống
        if not username or not password:
            flash('Vui lòng nhập đầy đủ tên người dùng và mật khẩu!', 'danger')
            return redirect(url_for('login'))

        # Kiểm tra thông tin đăng nhập
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]['password']):
            session['username'] = username
            session['role'] = users[username]['role']
            session['theme'] = users[username]['theme']
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('policy'))  # Chuyển hướng đến trang chính sách
        else:
            flash('Tên người dùng hoặc mật khẩu không đúng!', 'danger')

    return render_template('login.html', theme=session.get('theme', 'light'))

# Route cho trang chủ
@app.route('/')
def index():
    # Cho phép khách truy cập trang chủ
    if 'policy_agreed' not in session or not session['policy_agreed']:
        return redirect(url_for('policy'))
    role = session.get('role', 'guest')  # Nếu không đăng nhập, mặc định là khách
    return render_template('index.html', role=role, theme=session.get('theme', 'light'))

# Route cho trang hồ sơ
@app.route('/user_profile', methods=['GET', 'POST'])
def user_profile():
    if 'username' not in session:
        flash('Vui lòng đăng nhập để truy cập hồ sơ người dùng.', 'danger')
        return redirect(url_for('login'))
    if 'policy_agreed' not in session or not session['policy_agreed']:
        return redirect(url_for('policy'))
    if request.method == 'POST':
        if 'delete_history' in request.form:
            users[session['username']]['history'] = []
            save_users(users)
            flash('Lịch sử ẩn danh đã được xóa!', 'success')
        else:
            theme = request.form['theme']
            users[session['username']]['theme'] = theme
            session['theme'] = theme
            save_users(users)
            flash('Cài đặt giao diện đã được cập nhật!', 'success')
        return redirect(url_for('user_profile'))
    return render_template('user_profile.html', theme=session['theme'], history=users[session['username']]['history'])

# Route cho trang ẩn danh
@app.route('/anonymize', methods=['GET', 'POST'])
def anonymize():
    # Cho phép khách truy cập trang ẩn danh
    if 'policy_agreed' not in session or not session['policy_agreed']:
        return redirect(url_for('policy'))
    role = session.get('role', 'guest')
    if role not in ['user', 'guest']:
        flash('Bạn không có quyền truy cập trang này!', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        if 'cancel_anonymization' in request.form:
            if 'anonymized_file' in session:
                if os.path.exists(session['anonymized_file']):
                    os.remove(session['anonymized_file'])
                session.pop('anonymized_file', None)
                session.pop('results', None)
                session.pop('preview_data', None)
                session.pop('original_filename', None)  # Xóa tên file gốc
                flash('Đã hủy ẩn danh thành công!', 'success')
            return redirect(url_for('anonymize'))

        if 'reset_page' in request.form:
            if 'anonymized_file' in session and os.path.exists(session['anonymized_file']):
                os.remove(session['anonymized_file'])
            session.pop('anonymized_file', None)
            session.pop('results', None)
            session.pop('preview_data', None)
            session.pop('original_filename', None)  # Xóa tên file gốc
            flash('Đã reset trang thành công! Vui lòng chọn file mới.', 'success')
            return redirect(url_for('anonymize'))

        file = request.files['file']
        k_value = request.form['k_value']
        target_column = request.form['target_column'] if request.form['target_column'] else None

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Lưu tên file gốc vào session
            session['original_filename'] = filename

            try:
                anon_output_file, results = anonymize_data(filepath, int(k_value), target_column=target_column)
                session['anonymized_file'] = anon_output_file
                session['results'] = results

                # Đọc 5 dòng đầu tiên của file ẩn danh để xem trước
                df = pd.read_csv(anon_output_file, delimiter=';', nrows=5)
                preview_data = [df.columns.tolist()] + df.values.tolist()
                session['preview_data'] = preview_data

                # Lưu lịch sử ẩn danh chỉ cho người dùng đã đăng nhập
                if 'username' in session:
                    users[session['username']]['history'].append({
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'anonymized_info': f'File: {filename}, k={k_value}'
                    })
                    save_users(users)
                flash('Ẩn danh thành công!', 'success')
            except Exception as e:
                flash(f'Có lỗi xảy ra trong quá trình ẩn danh: {str(e)}', 'danger')
            return redirect(url_for('anonymize'))

    return render_template('anonymize.html', theme=session.get('theme', 'light'), results=session.get('results'), preview_data=session.get('preview_data'))

# Route để tải file ẩn danh
@app.route('/download_anonymized_file')
def download_anonymized_file():
    if 'anonymized_file' not in session:
        flash('Không có file ẩn danh để tải!', 'danger')
        return redirect(url_for('anonymize'))
    return send_file(session['anonymized_file'], as_attachment=True)

# Route để đăng xuất
@app.route('/logout')
def logout():
    session.clear()
    flash('Đã đăng xuất thành công!', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)