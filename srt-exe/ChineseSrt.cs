using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;

internal sealed class SubtitleEngine
{
    private static readonly string Runtime = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "Codex", "windows-chinese-srt");
    private readonly object sync = new object();
    private Process active;
    private bool cancelled;

    public bool Cancelled { get { lock (sync) return cancelled; } }

    public int Run(string source, string output, bool gpu, Action<string> log)
    {
        string python = Path.Combine(Runtime, "venv", "Scripts", "python.exe");
        if (!File.Exists(python) ||
            !File.Exists(Path.Combine(Runtime, "models", "whisper-small", "model.bin")) ||
            !File.Exists(Path.Combine(Runtime, "models", "nllb", "model.bin")))
            throw new InvalidOperationException("找不到本机的中文字幕运行环境或模型。请先安装 Codex 的 windows-chinese-srt 运行环境。");

        string temp = Path.Combine(Path.GetTempPath(), "ChineseSrt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temp);
        string script = Path.Combine(temp, "engine.py");
        try
        {
            using (Stream resource = Assembly.GetExecutingAssembly().GetManifestResourceStream("engine.py"))
            using (FileStream file = File.Create(script))
                resource.CopyTo(file);

            bool gpuError = false;
            int exitCode = RunOnce(python, script, source, output, gpu, line =>
            {
                if (Regex.IsMatch(line, @"out of memory|cuBLAS|cuDNN|CUDA (error|failed|driver)|no CUDA-capable", RegexOptions.IgnoreCase))
                    gpuError = true;
                log(line);
            });
            if (gpu && exitCode != 0 && gpuError && !Cancelled)
            {
                log("GPU 运行失败，正在切换到 CPU 继续。");
                return RunOnce(python, script, source, output, false, log);
            }
            return exitCode;
        }
        finally
        {
            if (File.Exists(script)) File.Delete(script);
            Directory.Delete(temp, false);
        }
    }

    private int RunOnce(string python, string script, string source, string output,
                        bool gpu, Action<string> log)
    {
        string device = gpu ? "cuda" : "cpu";
        var info = new ProcessStartInfo
        {
            FileName = python,
            Arguments = "-X utf8 -u " + Quote(script) + " " + Quote(source) + " " + Quote(output)
                + " --device " + device + " --translation-device " + device
                + " --batch-size " + (gpu ? "8" : "4")
                + " --cpu-threads " + (gpu ? "3" : "5") + " --retry-quiet",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        info.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
        using (var process = new Process { StartInfo = info })
        {
            process.OutputDataReceived += (sender, args) => { if (args.Data != null) log(args.Data); };
            process.ErrorDataReceived += (sender, args) => { if (args.Data != null) log(args.Data); };
            lock (sync)
            {
                if (cancelled) return -1;
                process.Start();
                active = process;
            }
            try
            {
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                process.WaitForExit();
                return process.ExitCode;
            }
            finally
            {
                lock (sync) active = null;
            }
        }
    }

    public void Cancel()
    {
        lock (sync)
        {
            cancelled = true;
            if (active == null || active.HasExited) return;
            var info = new ProcessStartInfo
            {
                FileName = Path.Combine(Environment.SystemDirectory, "taskkill.exe"),
                Arguments = "/PID " + active.Id + " /T /F",
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using (Process killer = Process.Start(info)) killer.WaitForExit();
        }
    }

    private static string Quote(string argument)
    {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in argument)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') result.Append('\\', slashes + 1);
            result.Append('\\', slashes).Append(c);
            slashes = 0;
        }
        return result.Append('\\', slashes * 2).Append('"').ToString();
    }
}

internal sealed class SubtitleForm : Form
{
    private readonly TextBox source = new TextBox();
    private readonly TextBox output = new TextBox();
    private readonly CheckBox gpu = new CheckBox();
    private readonly Button start = new Button();
    private readonly Button stop = new Button();
    private readonly Button open = new Button();
    private readonly RichTextBox log = new RichTextBox();
    private readonly Label status = new Label();
    private readonly ProgressBar progress = new ProgressBar();
    private readonly TableLayoutPanel inputs = new TableLayoutPanel();
    private SubtitleEngine engine;
    private int completed;
    private int total;

    public SubtitleForm()
    {
        Text = "简体中文 SRT 生成器";
        StartPosition = FormStartPosition.CenterScreen;
        ClientSize = new Size(800, 540);
        MinimumSize = new Size(720, 510);
        AutoScaleMode = AutoScaleMode.Dpi;
        Font = new Font("Microsoft YaHei UI", 9F);
        BackColor = Color.FromArgb(246, 248, 251);

        var layout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 6, Padding = new Padding(24, 19, 24, 20),
        };
        foreach (int height in new[] { 76, 86, 42, 26, 21 })
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        Controls.Add(layout);

        var header = new Panel { Dock = DockStyle.Fill };
        header.Controls.Add(new Label
        {
            Text = "简体中文 SRT 生成器", AutoSize = true, Location = new Point(0, 0),
            Font = new Font(Font.FontFamily, 21F, FontStyle.Bold), ForeColor = Color.FromArgb(20, 35, 53),
        });
        header.Controls.Add(new Label
        {
            Text = "本地转录与翻译 · 支持批量 · 输出独立字幕文件",
            AutoSize = true, Location = new Point(2, 42), ForeColor = Color.FromArgb(77, 96, 116),
        });
        layout.Controls.Add(header, 0, 0);

        inputs.Dock = DockStyle.Fill;
        inputs.ColumnCount = 4;
        inputs.RowCount = 2;
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 82));
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 88));
        inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 96));
        inputs.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        inputs.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        layout.Controls.Add(inputs, 0, 1);
        AddInput("输入媒体", source, 0);
        AddInput("字幕目录", output, 1);
        AddButton("选文件", 2, 0, PickFile);
        AddButton("选文件夹", 3, 0, () => PickFolder(true));
        AddButton("浏览", 2, 1, () => PickFolder(false));
        output.TextChanged += (sender, args) => open.Enabled = Directory.Exists(output.Text);

        var actions = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
        start.Text = "生成字幕";
        start.Size = new Size(102, 32);
        start.FlatStyle = FlatStyle.Flat;
        start.FlatAppearance.BorderSize = 0;
        start.BackColor = Color.FromArgb(21, 94, 167);
        start.ForeColor = Color.White;
        start.Click += Start;
        stop.Text = "停止";
        stop.Size = new Size(74, 32);
        stop.Enabled = false;
        stop.Click += Stop;
        open.Text = "打开字幕目录";
        open.Size = new Size(116, 32);
        open.Enabled = false;
        open.Click += (sender, args) =>
        {
            if (Directory.Exists(output.Text))
                Process.Start(new ProcessStartInfo { FileName = output.Text, UseShellExecute = true });
        };
        gpu.Text = "优先使用 NVIDIA GPU";
        gpu.Checked = true;
        gpu.AutoSize = true;
        gpu.Margin = new Padding(16, 10, 0, 0);
        actions.Controls.AddRange(new Control[] { start, stop, open, gpu });
        layout.Controls.Add(actions, 0, 2);

        status.Text = "请选择媒体文件或文件夹";
        status.Dock = DockStyle.Fill;
        status.TextAlign = ContentAlignment.MiddleLeft;
        layout.Controls.Add(status, 0, 3);
        progress.Dock = DockStyle.Fill;
        progress.Margin = new Padding(3, 2, 3, 5);
        layout.Controls.Add(progress, 0, 4);
        log.Dock = DockStyle.Fill;
        log.ReadOnly = true;
        log.BackColor = Color.White;
        log.ForeColor = Color.FromArgb(40, 56, 72);
        log.BorderStyle = BorderStyle.FixedSingle;
        log.DetectUrls = false;
        layout.Controls.Add(log, 0, 5);
        log.Text = "每份 SRT 与媒体同名。已有同名字幕会跳过。\n";

        AcceptButton = start;
        AllowDrop = true;
        DragEnter += (sender, args) =>
        {
            if (engine == null && args.Data.GetDataPresent(DataFormats.FileDrop))
                args.Effect = DragDropEffects.Copy;
        };
        DragDrop += (sender, args) =>
        {
            var paths = (string[])args.Data.GetData(DataFormats.FileDrop);
            if (engine == null && paths.Length > 0) SetSource(paths[0]);
        };
        FormClosing += (sender, args) =>
        {
            if (engine == null) return;
            args.Cancel = true;
            MessageBox.Show(this, "请先点击“停止”，等任务退出后再关闭。", Text,
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        };
    }

    private void AddInput(string caption, TextBox input, int row)
    {
        inputs.Controls.Add(new Label
        {
            Text = caption, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft,
        }, 0, row);
        input.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        inputs.Controls.Add(input, 1, row);
    }

    private void AddButton(string caption, int column, int row, Action action)
    {
        var button = new Button { Text = caption, Dock = DockStyle.Fill, Margin = new Padding(4, 7, 4, 7) };
        button.Click += (sender, args) => action();
        inputs.Controls.Add(button, column, row);
    }

    private void PickFile()
    {
        using (var dialog = new OpenFileDialog
        {
            Filter = "音频和视频|*.mp3;*.wav;*.m4a;*.aac;*.flac;*.ogg;*.opus;*.wma;*.mp4;*.mkv;*.mov;*.avi;*.webm;*.m4v;*.wmv;*.ts|所有文件|*.*",
            CheckFileExists = true,
        })
            if (dialog.ShowDialog(this) == DialogResult.OK) SetSource(dialog.FileName);
    }

    private void PickFolder(bool input)
    {
        using (var dialog = new FolderBrowserDialog { Description = input ? "选择媒体文件夹" : "选择字幕保存目录" })
            if (dialog.ShowDialog(this) == DialogResult.OK)
            {
                if (input) SetSource(dialog.SelectedPath);
                else output.Text = dialog.SelectedPath;
            }
    }

    private void SetSource(string path)
    {
        source.Text = path;
        output.Text = Directory.Exists(path) ? path : Path.GetDirectoryName(path);
    }

    private async void Start(object sender, EventArgs args)
    {
        string from = source.Text.Trim().Trim('"');
        string to = output.Text.Trim().Trim('"');
        if ((!File.Exists(from) && !Directory.Exists(from)) || String.IsNullOrWhiteSpace(to))
        {
            MessageBox.Show(this, "请选择有效的媒体文件或文件夹，并填写字幕保存目录。", Text,
                MessageBoxButtons.OK, MessageBoxIcon.Information);
            return;
        }
        var worker = new SubtitleEngine();
        engine = worker;
        bool useGpu = gpu.Checked;
        inputs.Enabled = gpu.Enabled = start.Enabled = false;
        stop.Enabled = true;
        progress.Value = completed = total = 0;
        log.Clear();
        status.Text = "正在加载本地模型…";
        try
        {
            int code = await Task.Run(() => worker.Run(from, to, useGpu, PostLine));
            status.Text = worker.Cancelled ? "已停止，已完成的字幕已保存"
                : code == 0 ? "处理完成" : "有文件未能生成字幕，请查看下方日志";
            if (code == 0 && !worker.Cancelled) progress.Value = progress.Maximum;
        }
        catch (Exception error)
        {
            status.Text = "无法完成";
            AppendLine(error.Message);
        }
        finally
        {
            engine = null;
            inputs.Enabled = gpu.Enabled = start.Enabled = true;
            stop.Enabled = false;
            open.Enabled = Directory.Exists(to);
        }
    }

    private async void Stop(object sender, EventArgs args)
    {
        SubtitleEngine worker = engine;
        if (worker == null) return;
        stop.Enabled = false;
        status.Text = "正在停止…";
        try { await Task.Run(() => worker.Cancel()); }
        catch (Exception error) { AppendLine(error.Message); stop.Enabled = engine != null; }
    }

    private void PostLine(string line)
    {
        if (!IsDisposed && IsHandleCreated)
            BeginInvoke((Action)(() => AppendLine(line)));
    }

    private void AppendLine(string line)
    {
        Match match = Regex.Match(line, @"^(\d+)/(\d+) (started|complete|already finished|failed)(.*)$");
        if (match.Success)
        {
            int index = Int32.Parse(match.Groups[1].Value);
            total = Int32.Parse(match.Groups[2].Value);
            string step = match.Groups[3].Value;
            if (step == "started")
            {
                status.Text = String.Format("正在处理第 {0} / {1} 个文件", index, total);
                line = status.Text + match.Groups[4].Value;
            }
            else
            {
                completed = Math.Max(completed, index);
                progress.Value = Math.Min(progress.Maximum, completed * progress.Maximum / total);
                line = String.Format("{0}/{1} {2}{3}", index, total,
                    step == "complete" ? "已生成" : step == "already finished" ? "已跳过同名字幕" : "未生成",
                    match.Groups[4].Value.Replace("No speech cues detected", "没有识别到可用对白"));
            }
        }
        else
        {
            Match audio = Regex.Match(line, @"^\s*(\d+)/(\d+) audio s, (\d+) cues");
            if (audio.Success && total > 0)
            {
                double duration = Math.Max(1, Int32.Parse(audio.Groups[2].Value));
                double fraction = Int32.Parse(audio.Groups[1].Value) / duration;
                progress.Value = Math.Min(progress.Maximum,
                    (int)((completed + fraction) * progress.Maximum / total));
                line = String.Format("  已扫描 {0}/{1} 秒，{2} 条字幕",
                    audio.Groups[1].Value, audio.Groups[2].Value, audio.Groups[3].Value);
            }
        }
        log.AppendText(line + Environment.NewLine);
        log.ScrollToCaret();
    }
}

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length == 4 && args[0] == "--self-test")
        {
            Directory.CreateDirectory(args[2]);
            using (var log = new StreamWriter(Path.Combine(args[2], "self-test.log"), false, Encoding.UTF8))
                return new SubtitleEngine().Run(args[1], args[2], args[3] == "cuda", line =>
                {
                    lock (log) { log.WriteLine(line); log.Flush(); }
                });
        }
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new SubtitleForm());
        return 0;
    }
}
