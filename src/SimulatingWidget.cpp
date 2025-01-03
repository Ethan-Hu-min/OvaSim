#include "SimulatingWidget.h"

SimulatingWidget::SimulatingWidget(QWidget* parent) : QWidget(parent)
{
	setAttribute(Qt::WA_DeleteOnClose);
	ui.setupUi(this);

	GLDisplayWidget = new GLWidget(ui.verticalLayoutWidget);
	GLDisplayWidget->setMinimumSize(QSize(800, 600));
	GLDisplayWidget->setMaximumSize(QSize(800, 600));
	ui.verticalLayout->addWidget(GLDisplayWidget);
	
	this->timer = new QTimer;
	connect(timer, SIGNAL(timeout()), this, SLOT(displayImage()));
	timer->start(33);
}

SimulatingWidget::~SimulatingWidget()
{
	qDebug() << "!!!!!!Simulating Widget Delete\n";
	delete timer;
	if(GLDisplayWidget != nullptr)delete GLDisplayWidget;
}

void SimulatingWidget::closeEvent(QCloseEvent*) {
	emit ExitWin();
}

void SimulatingWidget::displayImage() {
	GLDisplayWidget->setImage();
}
