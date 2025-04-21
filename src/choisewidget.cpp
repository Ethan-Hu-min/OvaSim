#include "choisewidget.h"


ChoiseWidget::ChoiseWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::ChoiseWidget)
{
    ui->setupUi(this);

    connect(ui->startButton, SIGNAL(clicked(bool)), this, SLOT(on_startbutton_clicked()));
    ui->sampleBox->addItem("案例 1");
    ui->sampleBox->addItem("案例 2");
    ui->sampleBox->addItem("案例 3");
    ui->sampleBox->addItem("案例 4");

    ui->modeBox->addItem("初学者模式");
    ui->modeBox->addItem("专业模式");

}

ChoiseWidget::~ChoiseWidget()
{
    delete ui;
}

void ChoiseWidget::getInfo(QString& name, QString& mode, QString& sample) {
    name = ui->lineEdit->text();
    mode = ui->sampleBox->currentText();
    sample = ui->modeBox->currentText();
}


void ChoiseWidget::on_startbutton_clicked() {

    emit ExitWin();
    this->close();
}
