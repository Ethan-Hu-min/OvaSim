#include "SimulatingWidget.h"
#include <GlobalConfig.h>

SimulatingWidget::SimulatingWidget(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_DeleteOnClose);
	ui.setupUi(this);
	ui.MessPrint->setReadOnly(true);
	ui.MessPrint->setLineWrapMode(QTextEdit::WidgetWidth);
	ui.MessPrint->setFont(QFont("Consolas", 10));
	ui.progressBar->setRange(0, 100);
	ui.progressBar->setValue(30);

	ui.progressBar->setTextVisible(false);

	GLDisplayWidget = new GLWidget(ui.verticalLayoutWidget);
	GLDisplayWidget->setMinimumSize(QSize(800, 600));
	GLDisplayWidget->setMaximumSize(QSize(800, 600));
	ui.verticalLayout->addWidget(GLDisplayWidget);
	
	connect(ui.SimStartButton, SIGNAL(clicked(bool)), this->GLDisplayWidget, SLOT(setStartRenderTrue()));
	connect(ui.SimStopButton, SIGNAL(clicked(bool)), this->GLDisplayWidget, SLOT(setStartRenderFalse()));

	connect(ui.SimStartButton, SIGNAL(clicked(bool)), this, SLOT(startTimer()));
	connect(ui.SimStopButton, SIGNAL(clicked(bool)), this, SLOT(endTimer()));

	connect(GLDisplayWidget, &GLWidget::fpsChanged, ui.fpsLabel, [this](int fps) { ui.fpsLabel->setNum(fps); });
	connect(GLDisplayWidget, &GLWidget::needleSwitchSignal, ui.NeedleSwitchLabel, [this](QString status) { ui.NeedleSwitchLabel->setText(status); });
	connect(GLDisplayWidget, &GLWidget::ovamNumsSignal, ui.OvamNuimsLabel, [this](int nums) { ui.OvamNuimsLabel->setNum(nums); });
	connect(GLDisplayWidget, &GLWidget::ovamNumsSignal, ui.OvamRateLabel, [this](int nums) { ui.OvamRateLabel->setText(QString("%1 %").arg( nums * 100.0 / GlobalConfig::OvamNums,0, 'f', 1)); });
	//connect(GLDisplayWidget, &GLWidget::nowCapacity, [this](int value) {ui.progressBar->setValue(value);});



	connect(GLDisplayWidget, SIGNAL(collideInfo(int)), this, SLOT(displayError(int)));

	this->timer = new QTimer;
	this->operateTimer = new QTimer;
	this->operateTimer->setInterval(500);
	connect(operateTimer, SIGNAL(timeout()), this, SLOT(displayTimer()));
	connect(timer, SIGNAL(timeout()), this->GLDisplayWidget, SLOT(setImage()));
	timer->start(int(1000/GlobalConfig::maxFps));
}

SimulatingWidget::~SimulatingWidget()
{
	qDebug() << "!!!!!!Simulating Widget Delete\n";
	if(timer != nullptr)delete timer;
	if (operateTimer != nullptr)delete operateTimer;
	if(GLDisplayWidget != nullptr)delete GLDisplayWidget;

}

void SimulatingWidget::startTimer() {
	if (isRunning) return;
	startTime = QDateTime::currentDateTime();
	operateTimer->start();
	isRunning = true;
}

void SimulatingWidget::endTimer() {
	if (!isRunning) return;
	operateTimer->stop();
	isRunning = false;
}


void SimulatingWidget::displayError(int infoType) {

	if (errorRecord.find(infoType) == errorRecord.end()) {
		QString msg;
		switch (infoType) {
		case 0: msg = "[警告]错误穿刺【膀胱】";break;
		case 1: msg = "[警告]错误穿刺【子宫】";break;
		case 2: msg = "[警告]错误穿刺【子宫】";break;
		case 3: msg = "[警告]错误穿刺【肠道】 ";break;
		case 10: msg = "[警告]错误吸取【不在卵泡内】";
			break;
		}
		errorRecord.insert(infoType);
		ui.MessPrint->append(msg);
	}
}

void SimulatingWidget::displayTimer() {
	qint64 elapsedMs = startTime.msecsTo(QDateTime::currentDateTime());
	int totalSeconds = static_cast<int>(elapsedMs / 1000);

	// 格式化为MM:SS
	int minutes = totalSeconds / 60;
	int seconds = totalSeconds % 60;
	QString timeText = QString("%1:%2")
		.arg(minutes, 2, 10, QChar('0'))
		.arg(seconds, 2, 10, QChar('0'));

	ui.OperatorTimeLabel->setText(timeText);
}

void SimulatingWidget::closeEvent(QCloseEvent*) {
	emit ExitWin();
}

void SimulatingWidget::keyPressEvent(QKeyEvent* event) {
	GLDisplayWidget->keyPressEvent(event);
}

void SimulatingWidget::keyReleaseEvent(QKeyEvent* event) {
	GLDisplayWidget->keyReleaseEvent(event);
}