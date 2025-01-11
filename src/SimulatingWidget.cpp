#include "SimulatingWidget.h"
#include <GlobalConfig.h>

SimulatingWidget::SimulatingWidget(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_DeleteOnClose);
	ui.setupUi(this);



	GLDisplayWidget = new GLWidget(ui.verticalLayoutWidget);
	GLDisplayWidget->setMinimumSize(QSize(800, 600));
	GLDisplayWidget->setMaximumSize(QSize(800, 600));
	ui.verticalLayout->addWidget(GLDisplayWidget);
	
	connect(ui.SimStartButton, SIGNAL(clicked(bool)), this->GLDisplayWidget, SLOT(setStartRenderTrue()));
	connect(ui.SimStopButton, SIGNAL(clicked(bool)), this->GLDisplayWidget, SLOT(setStartRenderFalse()));

	connect(GLDisplayWidget, &GLWidget::fpsChanged, ui.fpsLabel, [this](int fps) { ui.fpsLabel->setNum(fps); });
	connect(GLDisplayWidget, &GLWidget::needleSwitchSignal, ui.NeedleSwitchLabel, [this](QString status) { ui.NeedleSwitchLabel->setText(status); });
	connect(GLDisplayWidget, &GLWidget::ovamNumsSignal, ui.OvamNuimsLabel, [this](int nums) { ui.OvamNuimsLabel->setNum(nums); });

	this->timer = new QTimer;
	connect(timer, SIGNAL(timeout()), this->GLDisplayWidget, SLOT(setImage()));
	timer->start(int(1000/GlobalConfig::maxFps));
}

SimulatingWidget::~SimulatingWidget()
{
	qDebug() << "!!!!!!Simulating Widget Delete\n";
	if(timer != nullptr)delete timer;
	if(GLDisplayWidget != nullptr)delete GLDisplayWidget;

}

void SimulatingWidget::closeEvent(QCloseEvent*) {
	emit ExitWin();
}

void SimulatingWidget::keyPressEvent(QKeyEvent* event) {
	GLDisplayWidget->keyPressEvent(event);
}